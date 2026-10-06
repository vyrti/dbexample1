use std::fmt;

#[derive(Debug)]
pub enum Error {
    /// The SQL text does not parse.
    Parse(String),
    /// The SQL parses but refers to something that does not exist.
    Semantic(String),
    /// Valid SQL outside the subset this engine executes.
    Unsupported(String),
    /// A bound parameter is missing or of a type the engine cannot compare.
    Bind(String),
    /// SQLite's "integer overflow" from sum().
    IntegerOverflow,
    /// The file is not a valid sinew database.
    Format(String),
    Io(std::io::Error),
    #[cfg(feature = "sqlite-import")]
    Sqlite(rusqlite::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Parse(m) => write!(f, "parse error: {m}"),
            Error::Semantic(m) => write!(f, "{m}"),
            Error::Unsupported(m) => write!(f, "unsupported: {m}"),
            Error::Bind(m) => write!(f, "bind error: {m}"),
            Error::IntegerOverflow => write!(f, "integer overflow"),
            Error::Format(m) => write!(f, "bad sinew file: {m}"),
            Error::Io(e) => write!(f, "io: {e}"),
            #[cfg(feature = "sqlite-import")]
            Error::Sqlite(e) => write!(f, "sqlite: {e}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

#[cfg(feature = "sqlite-import")]
impl From<rusqlite::Error> for Error {
    fn from(e: rusqlite::Error) -> Self {
        Error::Sqlite(e)
    }
}
