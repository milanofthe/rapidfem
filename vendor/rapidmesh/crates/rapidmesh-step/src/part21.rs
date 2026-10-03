//! The exchange structure of ISO 10303-21: a STEP file as its entity
//! instances, each a record (a name and its parameters) or, for a complex
//! instance, several.

use rustc_hash::FxHashMap;

/// A parameter of a record.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    /// `#12`
    Ref(u32),
    Int(i64),
    Real(f64),
    Str(String),
    /// `.T.`, `.MILLI.` (without the dots)
    Enum(String),
    List(Vec<Value>),
    /// `LENGTH_MEASURE(1.E-07)`
    Typed(String, Box<Value>),
    /// `$`
    Unset,
    /// `*`
    Derived,
}

impl Value {
    pub fn as_ref(&self) -> Option<u32> {
        match self {
            Value::Ref(r) => Some(*r),
            _ => None,
        }
    }

    /// The number, an integer as well (`2` for `2.`).
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Real(x) => Some(*x),
            Value::Int(i) => Some(*i as f64),
            Value::Typed(_, v) => v.as_f64(),
            _ => None,
        }
    }

    pub fn as_int(&self) -> Option<i64> {
        match self {
            Value::Int(i) => Some(*i),
            _ => None,
        }
    }

    pub fn as_list(&self) -> Option<&[Value]> {
        match self {
            Value::List(l) => Some(l),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }

    /// `.T.` and `.F.`
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Enum(e) if e == "T" => Some(true),
            Value::Enum(e) if e == "F" => Some(false),
            _ => None,
        }
    }

    pub fn as_enum(&self) -> Option<&str> {
        match self {
            Value::Enum(e) => Some(e),
            _ => None,
        }
    }
}

/// One record: an entity name and its parameters.
#[derive(Clone, Debug, PartialEq)]
pub struct Record {
    pub name: String,
    pub args: Vec<Value>,
}

/// The instances of a file by id, each its records (one for a simple
/// instance, several for a complex one).
#[derive(Debug, Default)]
pub struct Exchange {
    pub schema: String,
    pub instances: FxHashMap<u32, Vec<Record>>,
}

impl Exchange {
    /// The record `name` of instance `id` (a part of a complex one as well).
    pub fn record(&self, id: u32, name: &str) -> Option<&Record> {
        self.instances.get(&id)?.iter().find(|r| r.name == name)
    }

    /// The name of instance `id` where it is simple.
    pub fn kind(&self, id: u32) -> Option<&str> {
        match self.instances.get(&id)?.as_slice() {
            [r] => Some(&r.name),
            _ => None,
        }
    }

    /// The ids of every instance holding a record `name`, ascending.
    pub fn all(&self, name: &str) -> Vec<u32> {
        let mut ids: Vec<u32> = self
            .instances
            .iter()
            .filter(|(_, rs)| rs.iter().any(|r| r.name == name))
            .map(|(&id, _)| id)
            .collect();
        ids.sort_unstable();
        ids
    }
}

/// Why a file does not read.
#[derive(Debug, Clone, PartialEq)]
pub struct ParseError {
    /// The byte offset where it stopped.
    pub at: usize,
    pub message: String,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "at byte {}: {}", self.at, self.message)
    }
}

impl std::error::Error for ParseError {}

struct Reader<'a> {
    s: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn fail<T>(&self, message: impl Into<String>) -> Result<T, ParseError> {
        Err(ParseError {
            at: self.at,
            message: message.into(),
        })
    }

    /// Skips white space and comments.
    fn skip(&mut self) {
        loop {
            while self.at < self.s.len() && self.s[self.at].is_ascii_whitespace() {
                self.at += 1;
            }
            if self.s[self.at..].starts_with(b"/*") {
                match self.s[self.at + 2..].windows(2).position(|w| w == b"*/") {
                    Some(k) => self.at += k + 4,
                    None => self.at = self.s.len(),
                }
            } else {
                return;
            }
        }
    }

    fn peek(&mut self) -> Option<u8> {
        self.skip();
        self.s.get(self.at).copied()
    }

    fn expect(&mut self, c: u8) -> Result<(), ParseError> {
        if self.peek() == Some(c) {
            self.at += 1;
            Ok(())
        } else {
            self.fail(format!("expected '{}'", c as char))
        }
    }

    fn keyword(&mut self) -> Result<String, ParseError> {
        self.skip();
        let from = self.at;
        while self.at < self.s.len()
            && (self.s[self.at].is_ascii_alphanumeric() || self.s[self.at] == b'_')
        {
            self.at += 1;
        }
        if self.at == from {
            return self.fail("expected a keyword");
        }
        Ok(String::from_utf8_lossy(&self.s[from..self.at]).into_owned())
    }

    fn number(&mut self) -> Result<Value, ParseError> {
        let from = self.at;
        if matches!(self.s.get(self.at), Some(b'+' | b'-')) {
            self.at += 1;
        }
        let mut real = false;
        while let Some(&c) = self.s.get(self.at) {
            match c {
                b'0'..=b'9' => {}
                b'.' | b'E' | b'e' => real = true,
                b'+' | b'-' if matches!(self.s[self.at - 1], b'E' | b'e') => {}
                _ => break,
            }
            self.at += 1;
        }
        let text = std::str::from_utf8(&self.s[from..self.at]).unwrap_or("");
        let parsed = if real {
            // `1.` and `1.E-07`: a trailing dot before the exponent is fine
            // for Rust's parser only without the exponent.
            text.replace(".E", ".0E")
                .replace(".e", ".0e")
                .parse::<f64>()
                .map(Value::Real)
                .ok()
        } else {
            text.parse::<i64>().map(Value::Int).ok()
        };
        match parsed {
            Some(v) => Ok(v),
            None => self.fail(format!("bad number {text:?}")),
        }
    }

    fn string(&mut self) -> Result<Value, ParseError> {
        self.at += 1;
        let mut out = Vec::new();
        loop {
            match self.s.get(self.at) {
                None => return self.fail("unterminated string"),
                Some(b'\'') if self.s.get(self.at + 1) == Some(&b'\'') => {
                    out.push(b'\'');
                    self.at += 2;
                }
                Some(b'\'') => {
                    self.at += 1;
                    return Ok(Value::Str(String::from_utf8_lossy(&out).into_owned()));
                }
                Some(&c) => {
                    out.push(c);
                    self.at += 1;
                }
            }
        }
    }

    fn value(&mut self) -> Result<Value, ParseError> {
        match self.peek() {
            Some(b'#') => {
                self.at += 1;
                match self.number()? {
                    Value::Int(i) if i >= 0 => Ok(Value::Ref(i as u32)),
                    _ => self.fail("bad reference"),
                }
            }
            Some(b'\'') => self.string(),
            Some(b'$') => {
                self.at += 1;
                Ok(Value::Unset)
            }
            Some(b'*') => {
                self.at += 1;
                Ok(Value::Derived)
            }
            Some(b'.') => {
                self.at += 1;
                let e = self.keyword()?;
                self.expect(b'.')?;
                Ok(Value::Enum(e))
            }
            Some(b'(') => Ok(Value::List(self.list()?)),
            Some(b'"') => {
                // A binary: kept as its hex digits.
                self.at += 1;
                let from = self.at;
                while self.s.get(self.at).is_some_and(|&c| c != b'"') {
                    self.at += 1;
                }
                let v = String::from_utf8_lossy(&self.s[from..self.at]).into_owned();
                self.at += 1;
                Ok(Value::Str(v))
            }
            Some(c) if c.is_ascii_digit() || c == b'-' || c == b'+' => self.number(),
            Some(c) if c.is_ascii_alphabetic() => {
                let name = self.keyword()?;
                let args = self.list()?;
                let inner = match args.len() {
                    1 => args.into_iter().next().unwrap_or(Value::Unset),
                    _ => Value::List(args),
                };
                Ok(Value::Typed(name, Box::new(inner)))
            }
            _ => self.fail("expected a value"),
        }
    }

    /// `( value, value, ... )`
    fn list(&mut self) -> Result<Vec<Value>, ParseError> {
        self.expect(b'(')?;
        let mut out = Vec::new();
        if self.peek() == Some(b')') {
            self.at += 1;
            return Ok(out);
        }
        loop {
            out.push(self.value()?);
            match self.peek() {
                Some(b',') => self.at += 1,
                Some(b')') => {
                    self.at += 1;
                    return Ok(out);
                }
                _ => return self.fail("expected ',' or ')'"),
            }
        }
    }

    /// `NAME(args)` or `( NAME(args) NAME(args) ... )`
    fn records(&mut self) -> Result<Vec<Record>, ParseError> {
        if self.peek() == Some(b'(') {
            self.at += 1;
            let mut out = Vec::new();
            while self.peek() != Some(b')') {
                let name = self.keyword()?;
                let args = self.list()?;
                out.push(Record { name, args });
            }
            self.at += 1;
            Ok(out)
        } else {
            let name = self.keyword()?;
            let args = self.list()?;
            Ok(vec![Record { name, args }])
        }
    }
}

/// Reads the exchange structure of `text`.
pub fn parse(text: &str) -> Result<Exchange, ParseError> {
    let mut r = Reader {
        s: text.as_bytes(),
        at: 0,
    };
    let mut out = Exchange::default();
    if let Some(k) = text.find("FILE_SCHEMA") {
        r.at = k;
        r.keyword()?;
        if let Ok(v) = r.list() {
            if let Some(Value::List(l)) = v.first() {
                out.schema = l.first().and_then(|s| s.as_str()).unwrap_or("").to_string();
            }
        }
    }
    let Some(data) = text.find("DATA;") else {
        return r.fail("no DATA section");
    };
    r.at = data + 5;
    loop {
        match r.peek() {
            Some(b'#') => {
                r.at += 1;
                let id = match r.number()? {
                    Value::Int(i) if i >= 0 => i as u32,
                    _ => return r.fail("bad instance id"),
                };
                r.expect(b'=')?;
                let records = r.records()?;
                r.expect(b';')?;
                out.instances.insert(id, records);
            }
            Some(_) => {
                let k = r.keyword()?;
                if k == "ENDSEC" {
                    return Ok(out);
                }
                return r.fail(format!("unexpected {k}"));
            }
            None => return r.fail("the DATA section does not end"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_complex_and_typed_instances_read() {
        let text = "ISO-10303-21;\nHEADER;\nFILE_SCHEMA(('AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }'));\nENDSEC;\nDATA;\n\
            #1 = CARTESIAN_POINT('',(0.,1.5,-2.E-03));\n\
            /* a comment */ #2 = ORIENTED_EDGE('',*,*,#3,.T.);\n\
            #4 = ( LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.) );\n\
            #5 = UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(1.E-07),#4,'it''s',$);\n\
            ENDSEC;\nEND-ISO-10303-21;\n";
        let x = parse(text).unwrap();
        assert!(x.schema.starts_with("AUTOMOTIVE_DESIGN"));
        let p = x.record(1, "CARTESIAN_POINT").unwrap();
        let c: Vec<f64> = p.args[1]
            .as_list()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        assert_eq!(c, vec![0.0, 1.5, -2e-3]);
        let e = x.record(2, "ORIENTED_EDGE").unwrap();
        assert_eq!(e.args[1], Value::Derived);
        assert_eq!(e.args[3].as_ref(), Some(3));
        assert_eq!(e.args[4].as_bool(), Some(true));
        assert_eq!(x.instances[&4].len(), 3);
        assert_eq!(
            x.record(4, "SI_UNIT").unwrap().args[0].as_enum(),
            Some("MILLI")
        );
        let u = x.record(5, "UNCERTAINTY_MEASURE_WITH_UNIT").unwrap();
        assert_eq!(u.args[0].as_f64(), Some(1e-7));
        assert_eq!(u.args[2].as_str(), Some("it's"));
        assert_eq!(u.args[3], Value::Unset);
        assert_eq!(x.kind(1), Some("CARTESIAN_POINT"));
        assert_eq!(x.kind(4), None);
    }
}
