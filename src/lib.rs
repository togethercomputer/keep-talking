use std::{borrow::Cow, cmp::Ordering, ops::Range};

use ahash::AHashMap;
use aho_corasick::{AhoCorasickBuilder, MatchKind};
use rayon::prelude::*;
use thiserror::Error;

#[cfg(feature = "pyo3")]
mod bindings;
pub mod loaders;
pub(crate) mod splitters;

#[cfg(feature = "pyo3")]
use pyo3::{prelude::*, types::PyModule};

use crate::splitters::{Split, Splitter, WordSplitter};

pub type Rank = u32;

struct EncoderEntry {
    priorities: Option<Box<[Option<Rank>]>>,
    rank: Rank,
}

type EncoderMap = AHashMap<Box<[u8]>, EncoderEntry>;
type SpecialEncoderMap = AHashMap<Box<[u8]>, Rank>;
type DecoderMap = Box<[Box<[u8]>]>;

pub struct Token {
    pub bytes: Vec<u8>,
    pub rank: Rank,
}

pub struct Tokenizer {
    encoder: EncoderMap,
    special_encoder: SpecialEncoderMap,
    decoder: DecoderMap,
    special_tokens_decoder: DecoderMap,
    prefix: Option<Vec<u8>>,
    splitters: Vec<Splitter>,
    word_splitter: WordSplitter,
}

impl Tokenizer {
    fn from_vocab_and_splitters(
        vocab: Vec<Token>,
        special_vocab: Vec<Token>,
        merges: Option<Vec<(Vec<u8>, Vec<u8>)>>,
        prefix: Option<Vec<u8>>,
        splitters: Vec<Splitter>,
        word_splitter: WordSplitter,
    ) -> Result<Self, Error> {
        let special_tokens_matcher = AhoCorasickBuilder::new()
            .match_kind(MatchKind::LeftmostLongest)
            .build(
                special_vocab
                    .iter()
                    .map(|v| v.bytes.as_slice())
                    .collect::<Vec<_>>(),
            )?;

        let mut splitters_acc = vec![Splitter::AhoCorasick(special_tokens_matcher)];
        splitters_acc.extend(splitters);

        let mut encoder = EncoderMap::default();
        let mut decoder_acc = Vec::new();

        for item in vocab.into_iter() {
            encoder.insert(
                item.bytes.clone().into_boxed_slice(),
                EncoderEntry {
                    rank: item.rank,
                    priorities: None,
                },
            );
            if decoder_acc.len() <= item.rank as usize {
                decoder_acc.resize(item.rank as usize + 1, Vec::new().into_boxed_slice());
            }
            decoder_acc[item.rank as usize] = item.bytes.into_boxed_slice();
        }
        let decoder: DecoderMap = decoder_acc.into_boxed_slice();

        if let Some(merges) = merges {
            merges
                .into_iter()
                .enumerate()
                .for_each(|(priority, (mut left, right))| {
                    let left_len = left.len();
                    left.extend_from_slice(&right);

                    encoder.get_mut(left.as_slice()).map(|entry| {
                        entry
                            .priorities
                            .get_or_insert(vec![None; left.len()].into_boxed_slice())[left_len] =
                            Some(priority as Rank);
                    });
                });
        };

        let mut special_encoder = SpecialEncoderMap::default();
        let mut special_tokens_decoder_acc = Vec::new();

        for item in special_vocab {
            special_encoder.insert(item.bytes.clone().into_boxed_slice(), item.rank);
            if special_tokens_decoder_acc.len() <= item.rank as usize {
                special_tokens_decoder_acc
                    .resize(item.rank as usize + 1, Vec::new().into_boxed_slice());
            }
            special_tokens_decoder_acc[item.rank as usize] = item.bytes.into_boxed_slice();
        }
        let special_tokens_decoder: DecoderMap = special_tokens_decoder_acc.into_boxed_slice();

        Ok(Self {
            encoder,
            special_encoder,
            decoder,
            special_tokens_decoder,
            prefix,
            splitters: splitters_acc,
            word_splitter,
        })
    }

    pub fn decode(&self, tokens: &[Rank]) -> Result<Vec<&[u8]>, Error> {
        let mut sequence = tokens
            .iter()
            .filter_map(|token| {
                if let Some(bytes) = self.decoder.get(*token as usize) {
                    Some(&**bytes)
                } else if let Some(bytes) = self.special_tokens_decoder.get(*token as usize) {
                    Some(&**bytes)
                } else {
                    return None;
                }
            })
            .collect::<Vec<_>>();

        if let Some(prefix) = &self.prefix {
            if sequence.first().map_or(false, |s| s.starts_with(prefix)) {
                sequence[0] = &sequence[0][prefix.len()..];
                return Ok(sequence);
            }
        }

        Ok(sequence)
    }

    pub fn decode_batch<T, I>(&self, tokens: T) -> Result<Vec<Vec<&[u8]>>, Error>
    where
        T: IntoParallelIterator<Item = I>,
        I: AsRef<[Rank]>,
    {
        tokens
            .into_par_iter()
            .map(|token| self.decode(token.as_ref()))
            .collect::<Result<Vec<_>, Error>>()
    }

    pub fn encode(&self, text: &[u8]) -> Result<Vec<Rank>, Error> {
        let text = if let Some(prefix) = &self.prefix {
            let mut text_acc = Vec::with_capacity(prefix.len() + text.len());
            text_acc.extend_from_slice(prefix);
            text_acc.extend_from_slice(text);
            Cow::Owned(text_acc)
        } else {
            Cow::Borrowed(text)
        };

        self.splitters
            .iter()
            .try_fold(
                {
                    let mut splits = Vec::with_capacity(text.len() / 4);
                    splits.push(Split::Bytes(0..text.len()));
                    splits
                },
                |splits, splitter| {
                    splits.into_iter().try_fold(
                        Vec::with_capacity(text.len() / 4),
                        |mut acc, split| {
                            match split {
                                Split::Bytes(r) => {
                                    splitter.split(&text[r.clone()], r.start, &mut acc)?
                                }
                                literal => acc.push(literal),
                            }

                            Ok::<_, Error>(acc)
                        },
                    )
                },
            )?
            .into_iter()
            .try_fold(Vec::with_capacity(text.len() / 4), |mut acc, split| {
                match split {
                    Split::Literal(r) => {
                        let bytes = &text[r.start..r.end];
                        if let Some(rank) = self.special_encoder.get(bytes) {
                            acc.push(*rank);
                        } else if let Some(entry) = self.encoder.get(bytes) {
                            acc.push(entry.rank);
                        } else {
                            return Err(Error::NoValidToken(
                                String::from_utf8_lossy(bytes).to_string(),
                            ));
                        }
                    }
                    Split::Bytes(r) => {
                        let bytes = &text[r.start..r.end];
                        if let Some(entry) = self.encoder.get(bytes) {
                            acc.push(entry.rank);
                        } else {
                            self.bpe_merge(bytes, &mut acc)?;
                        }
                    }
                }

                Ok(acc)
            })
    }

    pub fn encode_batch<T, I>(&self, texts: T) -> Result<Vec<Vec<Rank>>, Error>
    where
        T: IntoParallelIterator<Item = I>,
        I: AsRef<[u8]>,
    {
        texts
            .into_par_iter()
            .map(|text| self.encode(text.as_ref()))
            .collect()
    }

    fn bpe_merge(&self, chunk: &[u8], output: &mut Vec<Rank>) -> Result<(), Error> {
        let mut word_states = Vec::with_capacity(chunk.len());
        let mut matches = Vec::with_capacity(chunk.len());

        for (right_index, word_result) in self
            .word_splitter
            .into_iter(chunk, &self.encoder)
            .enumerate()
        {
            let (word, rank) = word_result?;
            word_states.push(WordState {
                rank,
                word,
                left_index: right_index.wrapping_sub(1),
                right_index,
                is_removed: false,
            });
        }

        for (left_index, window) in word_states.windows(2).enumerate() {
            let (WordState { word: left, .. }, WordState { word: right, .. }) =
                (&window[0], &window[1]);

            let combined = &chunk[left.start..right.end];
            let entry = self.encoder.get(combined);

            let (rank, priority, is_removed) = match entry {
                Some(entry) => (
                    entry.rank,
                    entry
                        .priorities
                        .as_ref()
                        .and_then(|p| p.get(left.len()).copied().flatten()),
                    false,
                ),
                None => (Rank::MAX, None, true),
            };

            matches.push(Match {
                word: left.start..right.end,
                left_index,
                right_index: left_index + 1,
                rank,
                priority,
                is_removed,
            });
        }

        loop {
            let Some((
                match_index,
                Match {
                    word,
                    rank,
                    left_index,
                    right_index,
                    ..
                },
            )) = matches
                .iter()
                .enumerate()
                .filter(|(_, m)| !m.is_removed)
                .min_by(|(_, l), (_, r)| match (l.priority, r.priority) {
                    (Some(left), Some(right)) => left.cmp(&right),
                    (Some(_), None) => Ordering::Less,
                    (None, Some(_)) => Ordering::Greater,
                    (None, None) => l.rank.cmp(&r.rank),
                })
                .map(|(index, m)| (index, m.clone()))
            else {
                break;
            };

            let new_word_state = &word_states[left_index];
            let consumed_word_state = &word_states[right_index];

            if let Some(left_match) = matches.get_mut(new_word_state.left_index) {
                let left_word = &word_states[left_match.left_index];
                let new_word = left_word.word.start..word.end;
                let new_entry = self.encoder.get(&chunk[new_word.clone()]);

                let (rank, priority, is_removed) = match new_entry {
                    Some(entry) => (
                        entry.rank,
                        entry
                            .priorities
                            .as_ref()
                            .and_then(|p| p[left_word.word.len()]),
                        false,
                    ),
                    None => (Rank::MAX, None, true),
                };

                left_match.word = new_word;
                left_match.rank = rank;
                left_match.priority = priority;
                left_match.is_removed = is_removed;
            }

            if let Some(right_match) = matches.get_mut(consumed_word_state.right_index) {
                let right_word = &word_states[right_match.right_index];
                let new_word = word.start..right_word.word.end;
                let new_entry = self.encoder.get(&chunk[new_word.clone()]);
                let (rank, priority, is_removed) = match new_entry {
                    Some(entry) => (
                        entry.rank,
                        entry.priorities.as_ref().and_then(|p| p[word.len()]),
                        false,
                    ),
                    None => (Rank::MAX, None, true),
                };

                right_match.word = new_word;
                right_match.rank = rank;
                right_match.priority = priority;
                right_match.is_removed = is_removed;
                right_match.left_index = left_index;
            }

            let new_right_match_index = consumed_word_state.right_index;
            let new_word_mut = &mut word_states[left_index];
            new_word_mut.right_index = new_right_match_index;
            new_word_mut.word = word;
            new_word_mut.rank = rank;

            word_states[right_index].is_removed = true;
            matches[match_index].is_removed = true;
        }

        output.extend(
            word_states
                .iter()
                .filter_map(|w| (!w.is_removed).then_some(w.rank)),
        );

        Ok(())
    }
}

struct WordState {
    word: Range<usize>,
    left_index: usize,
    right_index: usize,
    rank: Rank,
    is_removed: bool,
}

#[derive(Clone, Debug)]
struct Match {
    word: Range<usize>,
    left_index: usize,
    right_index: usize,
    rank: Rank,
    priority: Option<Rank>,
    is_removed: bool,
}

#[derive(Error, Debug)]
pub enum Error {
    #[error("regex compilation failed: {0}")]
    RegexError(#[from] pcre2::Error),

    #[error("aho-corasick build failed: {0}")]
    AhoCorasickError(#[from] aho_corasick::BuildError),

    #[error("invalid token for decoding: {0}")]
    InvalidToken(Rank),

    #[error("token not found in vocabulary: {0}")]
    NoValidToken(String),

    #[error("token not found in vocabulary during bpe encoding: {0:?}")]
    NoTokenForWord(Vec<u8>),

    #[error("invalid unicode sequence detected: {0:?}")]
    InvalidUnicodeSequence(Vec<u8>),

    #[error("token not found in vocabulary during prefix decoding: {0:?}")]
    NoPrefixToken(Vec<u8>),

    #[error("io error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("json parsing error: {0}")]
    JsonError(#[from] serde_json::Error),

    #[error("utf-8 decode error: {0}")]
    Utf8Error(#[from] std::str::Utf8Error),

    #[error("parse integer error: {0}")]
    ParseIntError(#[from] std::num::ParseIntError),

    #[error("utf-8 encode error: {0}")]
    Utf8EncodeError(#[from] std::string::FromUtf8Error),

    #[error("invalid model format: expected '{{token}} {{rank}}' per line")]
    InvalidModelFormat,

    #[error("unsupported normalizer and pretokenizer combination")]
    UnsupportedNormalizerPretokenizer,

    #[error("unexpected non-regex splitter pre-tokenizer")]
    UnexpectedNonRegexSplitter,

    #[error("missing supported normalizer and pretokenizer combination")]
    MissingNormalizerPretokenizer,
}

#[cfg(feature = "pyo3")]
#[pymodule]
fn keep_talkin(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<bindings::Tokenizer>()?;
    m.add_class::<bindings::Token>()?;
    bindings::register_exceptions(py, m)?;
    Ok(())
}
