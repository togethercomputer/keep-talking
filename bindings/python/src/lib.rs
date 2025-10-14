use keep_talkin::{Error, Rank};
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyType};

pyo3::create_exception!(keep_talkin, InitError, PyIOError);
pyo3::create_exception!(keep_talkin, EncodeError, PyValueError);
pyo3::create_exception!(keep_talkin, DecodeError, PyValueError);

#[pyclass(eq, hash, frozen)]
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Token {
    #[pyo3(get)]
    pub bytes: Vec<u8>,
    #[pyo3(get)]
    pub rank: Rank,
}

#[pymethods]
impl Token {
    #[new]
    fn new(bytes: Vec<u8>, rank: Rank) -> Self {
        Self { bytes, rank }
    }

    fn __repr__(&self) -> String {
        format!("Token(bytes={:?}, rank={})", self.bytes, self.rank)
    }

    fn __str__(&self) -> String {
        format!(
            "Token('{}', {})",
            String::from_utf8_lossy(&self.bytes),
            self.rank
        )
    }
}

impl From<keep_talkin::Token> for Token {
    fn from(token: keep_talkin::Token) -> Self {
        Self {
            bytes: token.bytes,
            rank: token.rank,
        }
    }
}

#[pyclass]
pub struct Tokenizer(keep_talkin::Tokenizer);

#[pymethods]
impl Tokenizer {
    #[classmethod]
    fn from_tokenizer_json(_cls: &Bound<'_, PyType>, path: &str) -> PyResult<Self> {
        let tokenizer = keep_talkin::Tokenizer::from_tokenizer_json(path)
            .map_err(|e| InitError::new_err(e.to_string()))?;

        Ok(Self(tokenizer))
    }

    #[classmethod]
    fn from_model_and_config(
        _cls: &Bound<'_, PyType>,
        model_path: &str,
        config_path: &str,
        regex_pattern: &str,
    ) -> PyResult<Self> {
        let tokenizer =
            keep_talkin::Tokenizer::from_model_and_config(model_path, config_path, [regex_pattern])
                .map_err(|e| InitError::new_err(e.to_string()))?;

        Ok(Self(tokenizer))
    }

    #[classmethod]
    fn from_tekken(_cls: &Bound<'_, PyType>, path: &str) -> PyResult<Self> {
        let tokenizer = keep_talkin::Tokenizer::from_tekken(path)
            .map_err(|e| InitError::new_err(e.to_string()))?;

        Ok(Self(tokenizer))
    }

    fn encode(&self, py: Python, data: &[u8]) -> PyResult<Vec<Rank>> {
        let Tokenizer(inner) = self;
        py.detach(|| {
            inner
                .encode(data)
                .map_err(|e| EncodeError::new_err(e.to_string()))
        })
    }

    fn decode(&self, py: Python, tokens: Vec<Rank>) -> PyResult<Py<PyAny>> {
        let Tokenizer(inner) = self;
        let bytes = py
            .detach(|| {
                let decoded = inner.decode(&tokens)?;
                Ok::<_, Error>(decoded.into_iter().flatten().copied().collect::<Vec<_>>())
            })
            .map_err(|e| DecodeError::new_err(e.to_string()))?;

        Ok(PyBytes::new(py, &bytes).into())
    }

    fn encode_batch(&self, py: Python, data: Vec<Vec<u8>>) -> PyResult<Vec<Vec<Rank>>> {
        let Tokenizer(inner) = self;
        py.detach(|| {
            inner
                .encode_batch(data)
                .map_err(|e| EncodeError::new_err(e.to_string()))
        })
    }

    fn decode_batch(&self, py: Python, tokens: Vec<Vec<Rank>>) -> PyResult<Vec<Py<PyAny>>> {
        let Tokenizer(inner) = self;
        let batch_bytes = py
            .detach(|| {
                let decoded = inner.decode_batch(tokens)?;

                Ok::<_, Error>(
                    decoded
                        .into_iter()
                        .map(|token_bytes| {
                            token_bytes
                                .into_iter()
                                .flatten()
                                .copied()
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>(),
                )
            })
            .map_err(|e| DecodeError::new_err(e.to_string()))?;

        Ok(batch_bytes
            .into_iter()
            .map(|bytes| PyBytes::new(py, &bytes).into())
            .collect())
    }
}

#[pymodule]
fn keep_talkin_py(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Tokenizer>()?;
    m.add_class::<Token>()?;
    m.add("InitError", py.get_type::<InitError>())?;
    m.add("EncodeError", py.get_type::<EncodeError>())?;
    m.add("DecodeError", py.get_type::<DecodeError>())?;
    Ok(())
}
