use nom::{
    branch::alt,
    bytes::complete::{is_not, tag},
    character::complete::{char, digit1},
    combinator::{map, opt, recognize},
    multi::{many0, many1},
    sequence::{delimited, pair, preceded, tuple},
    IResult,
};

pub fn comments(input: &str) -> IResult<&str, Vec<String>> {
    many0(alt((
        map(tag_parser, |s| s.to_string()),
        map(text, |s| s.to_string()),
    )))(input)
}

/// Parser for a tag
fn tag_parser(input: &str) -> IResult<&str, String> {
    delimited(
        tuple((char('['), char('%'))),
        alt((eval_parser, clk_parser)),
        char(']'),
    )(input)
}

/// Parser for an eval tag
fn eval_parser(input: &str) -> IResult<&str, String> {
    map(
        tuple((tag("eval"), spacing, signed_number)),
        |(_, _, value)| format!("[eval {}]", value),
    )(input)
}

/// Parser for a clk tag
fn clk_parser(input: &str) -> IResult<&str, String> {
    map(tuple((tag("clk"), spacing, time_value)), |(_, _, value)| {
        format!("[clk {}]", value)
    })(input)
}

/// Parser for a signed number
fn signed_number(input: &str) -> IResult<&str, String> {
    map(
        recognize(pair(
            opt(alt((char('+'), char('-')))),
            recognize(pair(digit1, opt(preceded(char('.'), digit1)))),
        )),
        |s: &str| s.to_string(),
    )(input)
}

/// Parser for a time value
fn time_value(input: &str) -> IResult<&str, String> {
    map(
        tuple((digit1, char(':'), digit1, char(':'), digit1)),
        |(h, _, m, _, s)| format!("{}:{}:{}", h, m, s),
    )(input)
}

/// Parser for text (any characters except '[' and ']')
fn text(input: &str) -> IResult<&str, &str> {
    is_not("[]")(input)
}

/// Parser for spacing (one or more spaces)
fn spacing(input: &str) -> IResult<&str, &str> {
    recognize(many1(char(' ')))(input)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_comments1() {
        let input = "[%eval 123] some text [%clk 12:34:56]";
        let result = comments(input);
        assert!(result.is_ok());
        let (_, parsed) = result.unwrap();
        assert_eq!(parsed, vec!["[eval 123]", " some text ", "[clk 12:34:56]"]);
    }

    #[test]
    fn test_comments2() {
        let input = "[%clk 12:34:56] some text ";
        let result = comments(input);
        assert!(result.is_ok());
        let (_, parsed) = result.unwrap();
        assert_eq!(parsed, vec!["[clk 12:34:56]", " some text "]);
    }

    #[test]
    fn test_tag_parser() {
        let input = "[%eval 123]";
        let result = tag_parser(input);
        assert!(result.is_ok());
        let (_, parsed) = result.unwrap();
        assert_eq!(parsed, "[eval 123]");
    }

    #[test]
    fn test_tag_parser_incorrect_name() {
        let input = "[%clk 123]";
        let result = tag_parser(input);
        assert!(result.is_err());
    }

    #[test]
    fn test_clk_parser() {
        let input = "clk 12:34:56";
        let result = clk_parser(input);
        assert!(result.is_ok());
        let (_, parsed) = result.unwrap();
        assert_eq!(parsed, "[clk 12:34:56]");
    }
    #[test]
    fn test_clk_parser_incorrect_name() {
        let input = "eval 123";
        let result = clk_parser(input);
        assert!(result.is_err());
    }

    #[test]
    fn test_text() {
        let input = "some text";
        let result = text(input);
        assert!(result.is_ok());
        let (_, parsed) = result.unwrap();
        assert_eq!(parsed, "some text");
    }

    #[test]
    fn test_signed_number() {
        let input = "-123.45";
        let result = signed_number(input);
        assert!(result.is_ok());
        let (_, parsed) = result.unwrap();
        assert_eq!(parsed, "-123.45");
    }

    #[test]
    fn test_time_value() {
        let input = "12:34:56";
        let result = time_value(input);
        assert!(result.is_ok());
        let (_, parsed) = result.unwrap();
        assert_eq!(parsed, "12:34:56");
    }
}

// Example usage
fn main() {
    let input = "   [%eval 123] some text [%clk +12:34:56]  ";
    match comments(input) {
        Ok((remaining, parsed)) => {
            println!("Parsed: {:?}", parsed);
            println!("Remaining: {:?}", remaining);
        }
        Err(err) => {
            println!("Error: {:?}", err);
        }
    }
}
