use std::borrow::Cow;
use std::sync::Arc;
use std::time::Duration;

use bincode::{DefaultOptions, Options};
use bytes::{Bytes, BytesMut};
use futures::Stream;
use futures::StreamExt;
use futures::stream::{SplitSink, SplitStream};
use futures::{Sink, SinkExt};
use orion::aead::streaming::{Nonce, StreamOpener, StreamSealer, StreamTag};
use orion::kdf::SecretKey;
use orion::util::secure_rand_bytes;
use serde::Deserialize;
use serde::de::DeserializeOwned;
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::sync::mpsc::UnboundedReceiver;
use tokio::time::timeout;
use tokio_util::codec::{Framed, LengthDelimitedCodec};

/// Maximum amount of actual message data placed in a single wire frame (i.e. a
/// single write to the underlying TCP socket). Vista's network path between login
/// and `gg`-partition compute nodes silently and permanently black-holes a TCP
/// connection the moment a single write needs more than one TCP segment (observed
/// threshold: writes up to ~1460 bytes -- the standard MSS for a 1500-byte-MTU path
/// -- always succeed; anything larger fails 100% of the time and kills the
/// connection outright, with no further data ever delivered in either direction).
/// Splitting any larger message into several sends of at most this size, each
/// safely fitting in one segment with margin to spare, avoids the black hole
/// entirely. See CHUNK_CONTINUES/CHUNK_FINAL below for the wire format.
///
/// Vista-specific: not observed on this fork's x86 sites, and gated to aarch64
/// (Vista is currently the only aarch64 deployment) -- see send_chunked/
/// read_reassembled_message, the only place this is used.
#[cfg(target_arch = "aarch64")]
const MAX_WIRE_CHUNK_PAYLOAD: usize = 1200;

/// Prefix byte on every wire frame: CHUNK_FINAL means this frame completes the
/// logical message (the message may consist of just this one frame); CHUNK_CONTINUES
/// means more frames follow before the logical message is complete.
#[cfg(target_arch = "aarch64")]
const CHUNK_CONTINUES: u8 = 1;
#[cfg(target_arch = "aarch64")]
const CHUNK_FINAL: u8 = 0;

/// Real (not just scheduler-yield) pause between consecutive chunks of the
/// same oversized message, so the kernel/NIC has a chance to actually
/// transmit each chunk as its own packet before the next is queued -- see the
/// comment at its use site in send_chunked for why this is necessary even
/// though each chunk is already sized to fit in one TCP segment on its own.
#[cfg(target_arch = "aarch64")]
const CHUNK_SEND_DELAY: Duration = Duration::from_millis(2);

use crate::internal::common::error::DsError;
use crate::internal::messages::auth::{
    AuthenticationError, AuthenticationMode, AuthenticationRequest, AuthenticationResponse,
    Challenge, EncryptionResponse,
};

const CHALLENGE_LENGTH: usize = 16;

pub(crate) struct Authenticator {
    pub(crate) protocol: u32,
    pub(crate) my_role: &'static str,
    pub(crate) peer_role: &'static str,
    pub(crate) secret_key: Option<Arc<SecretKey>>,
    pub(crate) challenge: Vec<u8>,
    pub(crate) sealer: Option<StreamSealer>,
    pub(crate) error: Option<String>,
}

impl Authenticator {
    pub fn new(
        protocol: u32,
        role: &'static str,
        peer_role: &'static str,
        secret_key: Option<Arc<SecretKey>>,
    ) -> Self {
        Authenticator {
            protocol,
            my_role: role,
            peer_role,
            secret_key,
            challenge: Default::default(),
            sealer: None,
            error: None,
        }
    }

    pub fn make_auth_request(&mut self) -> crate::Result<AuthenticationRequest> {
        let mode = if self.secret_key.is_some() {
            let mut challenge = vec![0; CHALLENGE_LENGTH];
            secure_rand_bytes(&mut challenge).map_err(|_| "Generating challenge failed")?;
            self.challenge.clone_from(&challenge);
            AuthenticationMode::Encryption(Challenge { challenge })
        } else {
            AuthenticationMode::NoAuth
        };
        Ok(AuthenticationRequest {
            protocol: self.protocol,
            role: Cow::Borrowed(self.my_role),
            mode,
        })
    }

    pub(crate) fn _make_error(&mut self, message: String) -> crate::Result<AuthenticationResponse> {
        self.error = Some(message.clone());
        Ok(AuthenticationResponse::Error(AuthenticationError {
            message,
        }))
    }

    pub fn make_auth_response(
        &mut self,
        message: AuthenticationRequest,
    ) -> crate::Result<AuthenticationResponse> {
        if message.protocol != self.protocol {
            return self._make_error(format!(
                "Invalid version of protocol, expected {}, got {}",
                self.protocol, message.protocol
            ));
        }

        if message.role != self.peer_role {
            return self._make_error(format!(
                "Expected peer role {}, got {}",
                self.peer_role, message.role
            ));
        }

        match &mut (message.mode, &self.secret_key) {
            (AuthenticationMode::NoAuth, None) => Ok(AuthenticationResponse::NoAuth),
            (AuthenticationMode::Encryption(msg), Some(key)) => {
                log::trace!("Peer authorization started");
                if msg.challenge.len() != CHALLENGE_LENGTH {
                    return self._make_error(format!(
                        "Invalid length of challenge ({})",
                        msg.challenge.len()
                    ));
                }

                let (mut sealer, nonce) =
                    StreamSealer::new(key).map_err(|_| "Creating sealer failed")?;

                let mut response = Vec::new();
                response.extend_from_slice(self.my_role.as_bytes());
                response.extend_from_slice(&msg.challenge);

                let challenge_response = sealer
                    .seal_chunk(&response, &StreamTag::Message)
                    .map_err(|_| "Cannot seal challenge")?;
                self.sealer = Some(sealer);

                Ok(AuthenticationResponse::Encryption(EncryptionResponse {
                    nonce: nonce.as_ref().into(),
                    response: challenge_response,
                }))
            }
            (AuthenticationMode::Encryption(_), None) => {
                self._make_error("Peer requests authentication".to_string())
            }
            (AuthenticationMode::NoAuth, Some(_)) => {
                self._make_error("Peer does not support authentication".to_string())
            }
        }
    }

    pub fn finish_authentication(
        mut self,
        message: AuthenticationResponse,
    ) -> crate::Result<(Option<StreamSealer>, Option<StreamOpener>)> {
        if let Some(error) = std::mem::take(&mut self.error) {
            return Err(DsError::AuthenticationRejected(format!(
                "Authentication failed: {error}"
            )));
        }

        let opener = match (message, &self.secret_key) {
            (AuthenticationResponse::Error(error), _) => {
                return Err(DsError::AuthenticationRejected(format!(
                    "Received authentication error: {}",
                    error.message
                )));
            }
            (AuthenticationResponse::NoAuth, None) => {
                log::trace!("Empty authentication finished");
                None
            }
            (AuthenticationResponse::Encryption(response), Some(key)) => {
                log::trace!("Challenge verification started");
                let remote_nonce = &Nonce::from_slice(&response.nonce)
                    .map_err(|_| DsError::AuthenticationRejected("Invalid nonce".to_string()))?;
                let mut opener = StreamOpener::new(key, remote_nonce).map_err(|_| {
                    DsError::AuthenticationRejected("Failed to create opener".to_string())
                })?;
                let (opened_challenge, tag) =
                    opener.open_chunk(&response.response).map_err(|_| {
                        DsError::AuthenticationRejected("Cannot verify challenge".to_string())
                    })?;

                let mut expected_response = Vec::new();
                expected_response.extend_from_slice(self.peer_role.as_bytes());
                expected_response.extend_from_slice(&self.challenge);

                if tag != StreamTag::Message || opened_challenge != expected_response {
                    return Err(DsError::AuthenticationRejected(
                        "Received challenge does not match.".to_string(),
                    ));
                }
                log::trace!("Challenge verification finished");
                Some(opener)
            }
            (_, _) => {
                return Err(DsError::AuthenticationRejected(
                    "Invalid authentication state".to_string(),
                ));
            }
        };
        Ok((self.sealer, opener))
    }
}

pub async fn do_authentication<T: AsyncRead + AsyncWrite>(
    protocol: u32,
    my_role: &'static str,
    peer_role: &'static str,
    secret_key: Option<Arc<SecretKey>>,
    writer: &mut SplitSink<Framed<T, LengthDelimitedCodec>, bytes::Bytes>,
    reader: &mut SplitStream<Framed<T, LengthDelimitedCodec>>,
) -> crate::Result<(Option<StreamSealer>, Option<StreamOpener>)> {
    const AUTH_TIMEOUT: Duration = Duration::from_secs(15);
    let mut authenticator = Authenticator::new(protocol, my_role, peer_role, secret_key);

    /* Send authentication message */
    let message = authenticator.make_auth_request()?;
    let message_data = serialize(&message).unwrap().into();
    timeout(AUTH_TIMEOUT, writer.send(message_data))
        .await
        .map_err(|_| "Sending authentication timeout")?
        .map_err(|_| "Sending authentication failed")?;

    /* Receive authentication message */
    let remote_message_data = timeout(AUTH_TIMEOUT, reader.next())
        .await
        .map_err(|_| "Authentication message did not arrived")?
        .ok_or_else(|| {
            DsError::from("The remote side closed connection without authentication message")
        })??;
    let remote_message: AuthenticationRequest = deserialize(&remote_message_data)?;

    /* Send authentication response */
    let response = authenticator.make_auth_response(remote_message)?;
    let response_data = serialize(&response).unwrap().into();
    timeout(AUTH_TIMEOUT, writer.send(response_data))
        .await
        .map_err(|_| "Sending authentication timeouted")?
        .map_err(|_| "Sending authentication failed")?;

    /* Receive authentication response */
    let remote_response_data = timeout(AUTH_TIMEOUT, reader.next())
        .await
        .map_err(|_| "Authentication message did not arrived")?
        .ok_or_else(|| {
            DsError::from("The remote side closed connection without authentication message")
        })??;
    let remote_response: AuthenticationResponse = deserialize(&remote_response_data)?;

    // Finish authentication
    authenticator.finish_authentication(remote_response)
}

pub fn open_message<T>(opener: &mut Option<StreamOpener>, message_data: &[u8]) -> crate::Result<T>
where
    T: DeserializeOwned,
{
    if let Some(opener) = opener {
        let (msg, tag) = opener
            .open_chunk(message_data)
            .map_err(|_| DsError::GenericError("Cannot decrypt message".to_string()))?;
        assert_eq!(tag, StreamTag::Message);
        Ok(deserialize(&msg)?)
    } else {
        Ok(deserialize(message_data)?)
    }
}

#[inline]
pub fn serialize<T>(value: &T) -> crate::Result<Vec<u8>>
where
    T: serde::Serialize + ?Sized,
{
    DefaultOptions::new()
        .with_limit(crate::MAX_FRAME_SIZE as u64)
        .with_fixint_encoding()
        .serialize(value)
        .map_err(|e| format!("Serialization failed: {e:?}").into())
}

#[inline]
pub fn deserialize<'a, T>(bytes: &'a [u8]) -> crate::Result<T>
where
    T: Deserialize<'a>,
{
    DefaultOptions::new()
        .with_limit(crate::MAX_FRAME_SIZE as u64)
        .with_fixint_encoding()
        .deserialize(bytes)
        .map_err(|e| format!("Deserialization failed: {e:?}, data {bytes:?}").into())
}

#[inline]
pub fn seal_message(sealer: &mut Option<StreamSealer>, data: Bytes) -> Bytes {
    if let Some(sealer) = sealer {
        sealer
            .seal_chunk(&data, &StreamTag::Message)
            .unwrap()
            .into()
    } else {
        data
    }
}

pub async fn forward_queue_to_sealed_sink<E, S: Sink<Bytes, Error = E> + Unpin>(
    mut queue: UnboundedReceiver<Bytes>,
    mut sink: S,
    mut sealer: Option<StreamSealer>,
) -> Result<(), E> {
    while let Some(data) = queue.recv().await {
        let sealed = seal_message(&mut sealer, data);
        if let Err(e) = send_chunked(&mut sink, sealed).await {
            log::debug!("Forwarding from queue failed");
            return Err(e);
        }
    }
    Ok(())
}

/// Sends `data` as one or more wire frames, each carrying at most
/// `MAX_WIRE_CHUNK_PAYLOAD` bytes of the message so no single write ever needs
/// more than one TCP segment on a standard-MTU path (see MAX_WIRE_CHUNK_PAYLOAD).
///
/// Vista-only; every other site's build takes the plain, unchunked send below,
/// byte-identical to this fork's pre-chunking wire format.
#[cfg(target_arch = "aarch64")]
pub async fn send_chunked<E, S: Sink<Bytes, Error = E> + Unpin>(
    sink: &mut S,
    data: Bytes,
) -> Result<(), E> {
    if data.len() <= MAX_WIRE_CHUNK_PAYLOAD {
        let mut frame = BytesMut::with_capacity(data.len() + 1);
        frame.extend_from_slice(&[CHUNK_FINAL]);
        frame.extend_from_slice(&data);
        return sink.send(frame.freeze()).await;
    }
    let mut offset = 0;
    while offset < data.len() {
        let end = (offset + MAX_WIRE_CHUNK_PAYLOAD).min(data.len());
        let is_last = end == data.len();
        let mut frame = BytesMut::with_capacity(end - offset + 1);
        frame.extend_from_slice(&[if is_last { CHUNK_FINAL } else { CHUNK_CONTINUES }]);
        frame.extend_from_slice(&data[offset..end]);
        sink.send(frame.freeze()).await?;
        offset = end;
        if offset < data.len() {
            // Without a real pause here, many chunks queued back-to-back can be
            // recombined by the kernel/NIC (TCP segmentation offload) into fewer,
            // larger on-wire packets that exceed the safe single-segment size
            // again, defeating the whole point of chunking. A short sleep gives
            // the previous chunk time to actually leave the NIC as its own
            // packet before the next one is queued.
            tokio::time::sleep(CHUNK_SEND_DELAY).await;
        }
    }
    Ok(())
}

#[cfg(not(target_arch = "aarch64"))]
pub async fn send_chunked<E, S: Sink<Bytes, Error = E> + Unpin>(
    sink: &mut S,
    data: Bytes,
) -> Result<(), E> {
    sink.send(data).await
}

/// Reads wire frames from `stream` and reassembles them into one complete
/// logical message, transparently undoing the chunking done by `send_chunked`.
/// Returns `Ok(None)` if the stream ended before any (partial) message data
/// was read, matching the behavior of `stream.next()` at end-of-stream.
///
/// Vista-only; every other site's build takes the plain passthrough below, which
/// reads exactly one frame per message, matching this fork's pre-chunking behavior.
#[cfg(target_arch = "aarch64")]
pub async fn read_reassembled_message<S>(
    stream: &mut S,
) -> Result<Option<BytesMut>, std::io::Error>
where
    S: Stream<Item = Result<BytesMut, std::io::Error>> + Unpin,
{
    let mut accumulated: Option<BytesMut> = None;
    loop {
        let Some(frame) = stream.next().await else {
            return Ok(None);
        };
        let mut frame = frame?;
        if frame.is_empty() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "received an empty chunked-message frame (missing continuation marker)",
            ));
        }
        let marker = frame.split_to(1)[0];
        match accumulated.as_mut() {
            Some(buf) => buf.unsplit(frame),
            None => accumulated = Some(frame),
        }
        if marker == CHUNK_FINAL {
            return Ok(accumulated);
        }
        // marker == CHUNK_CONTINUES: loop to read the next frame of this message.
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub async fn read_reassembled_message<S>(
    stream: &mut S,
) -> Result<Option<BytesMut>, std::io::Error>
where
    S: Stream<Item = Result<BytesMut, std::io::Error>> + Unpin,
{
    stream.next().await.transpose()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use orion::kdf::SecretKey;

    use crate::internal::messages::auth::AuthenticationResponse;
    use crate::internal::transfer::auth::Authenticator;

    #[test]
    fn test_no_auth() {
        let mut a1 = Authenticator::new(0, "a", "b", None);
        let mut a2 = Authenticator::new(0, "b", "a", None);

        let q1 = a1.make_auth_request().unwrap();
        let q2 = a2.make_auth_request().unwrap();

        let r1 = a1.make_auth_response(q2).unwrap();
        let r2 = a2.make_auth_response(q1).unwrap();

        assert!(matches!(&r1, AuthenticationResponse::NoAuth));
        assert!(matches!(&r2, AuthenticationResponse::NoAuth));

        let (s1, o1) = a1.finish_authentication(r2).unwrap();
        let (s2, o2) = a2.finish_authentication(r1).unwrap();

        assert!(s1.is_none());
        assert!(s2.is_none());
        assert!(o1.is_none());
        assert!(o2.is_none());
    }

    #[test]
    fn test_auth_ok() {
        let secret_key = Some(Arc::new(SecretKey::generate(32).unwrap()));
        let mut a1 = Authenticator::new(0, "a", "b", secret_key.clone());
        let mut a2 = Authenticator::new(0, "b", "a", secret_key);

        let q1 = a1.make_auth_request().unwrap();
        let q2 = a2.make_auth_request().unwrap();

        let r1 = a1.make_auth_response(q2).unwrap();
        let r2 = a2.make_auth_response(q1).unwrap();

        assert!(matches!(&r1, AuthenticationResponse::Encryption(_)));
        assert!(matches!(&r2, AuthenticationResponse::Encryption(_)));

        let (s1, o1) = a1.finish_authentication(r2).unwrap();
        let (s2, o2) = a2.finish_authentication(r1).unwrap();

        assert!(s1.is_some());
        assert!(s2.is_some());
        assert!(o1.is_some());
        assert!(o2.is_some());
    }

    #[test]
    fn test_auth_different_keys() {
        let secret_key1 = Some(Arc::new(SecretKey::generate(32).unwrap()));
        let secret_key2 = Some(Arc::new(SecretKey::generate(32).unwrap()));
        let mut a1 = Authenticator::new(0, "a", "b", secret_key1);
        let mut a2 = Authenticator::new(0, "b", "a", secret_key2);

        let q1 = a1.make_auth_request().unwrap();
        let q2 = a2.make_auth_request().unwrap();

        let r1 = a1.make_auth_response(q2).unwrap();
        let r2 = a2.make_auth_response(q1).unwrap();

        assert!(matches!(&r1, AuthenticationResponse::Encryption(_)));
        assert!(matches!(&r2, AuthenticationResponse::Encryption(_)));

        assert!(a1.finish_authentication(r2).is_err());
        assert!(a2.finish_authentication(r1).is_err());
    }

    #[test]
    fn test_auth_and_no_auth() {
        let secret_key = Some(Arc::new(SecretKey::generate(32).unwrap()));
        let mut a1 = Authenticator::new(0, "a", "b", secret_key);
        let mut a2 = Authenticator::new(0, "b", "a", None);

        let q1 = a1.make_auth_request().unwrap();
        let q2 = a2.make_auth_request().unwrap();

        let r1 = a1.make_auth_response(q2).unwrap();
        let r2 = a2.make_auth_response(q1).unwrap();

        assert!(matches!(&r1, AuthenticationResponse::Error(_)));
        assert!(matches!(&r2, AuthenticationResponse::Error(_)));

        assert!(a1.finish_authentication(r2).is_err());
        assert!(a2.finish_authentication(r1).is_err());
    }

    #[test]
    fn test_mirror_attack() {
        let secret_key = Some(Arc::new(SecretKey::generate(32).unwrap()));
        let mut a1 = Authenticator::new(0, "a", "b", secret_key);

        let mut q1 = a1.make_auth_request().unwrap();
        q1.role = "b".into();
        let r1 = a1.make_auth_response(q1).unwrap();
        assert!(a1.finish_authentication(r1).is_err());
    }

    #[test]
    fn test_invalid_version() {
        let mut a1 = Authenticator::new(0, "a", "b", None);
        let mut a2 = Authenticator::new(1, "b", "a", None);

        let q1 = a1.make_auth_request().unwrap();
        let q2 = a2.make_auth_request().unwrap();

        let r1 = a1.make_auth_response(q2).unwrap();
        let r2 = a2.make_auth_response(q1).unwrap();

        assert!(matches!(&r1, AuthenticationResponse::Error(_)));
        assert!(matches!(&r2, AuthenticationResponse::Error(_)));

        assert!(a1.finish_authentication(r2).is_err());
        assert!(a2.finish_authentication(r1).is_err());
    }

    #[test]
    fn test_invalid_roles() {
        let mut a1 = Authenticator::new(0, "a", "b", None);
        let mut a2 = Authenticator::new(0, "b", "c", None);

        let q1 = a1.make_auth_request().unwrap();
        let q2 = a2.make_auth_request().unwrap();

        let r1 = a1.make_auth_response(q2).unwrap();
        let r2 = a2.make_auth_response(q1).unwrap();

        assert!(matches!(&r1, AuthenticationResponse::NoAuth));
        assert!(matches!(&r2, AuthenticationResponse::Error(_)));

        assert!(a1.finish_authentication(r2).is_err());
        assert!(a2.finish_authentication(r1).is_err());
    }
}
