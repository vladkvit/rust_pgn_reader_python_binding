use nom::{
    branch::alt,
    bytes::complete::{is_not, tag},
    character::complete::{char, digit1},
    combinator::{map, opt, recognize},
    multi::{many0, many1},
    sequence::{delimited, pair, preceded},
    IResult, Parser,
};

#[derive(Debug, PartialEq)]
pub enum CommentContent {
    Text(String),
    Eval(f64),
    ClkTime((u32, u8, f64)),
}

pub fn parse_comments(input: &str) -> IResult<&str, Vec<CommentContent>> {
    many0(alt((
        map(tag_parser, |s| match s.as_str() {
            eval if eval.starts_with("[eval ") => {
                let value = &eval[6..eval.len() - 1];
                CommentContent::Eval(value.parse().unwrap_or_default())
            }
            clk if clk.starts_with("[clk ") => {
                let time_parts: Vec<&str> = clk[5..clk.len() - 1].split(':').collect();
                let hours = time_parts
                    .get(0)
                    .and_then(|h| h.parse().ok())
                    .unwrap_or_default();
                let minutes = time_parts
                    .get(1)
                    .and_then(|m| m.parse().ok())
                    .unwrap_or_default();
                let seconds = time_parts
                    .get(2)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or_default();
                CommentContent::ClkTime((hours, minutes, seconds))
            }
            _ => unreachable!(),
        }),
        map(text, |s| CommentContent::Text(s.to_string())),
    )))
    .parse(input)
}

/// Parser for a tag
fn tag_parser(input: &str) -> IResult<&str, String> {
    delimited(
        (char('['), char('%')),
        alt((eval_parser, clk_parser)),
        char(']'),
    )
    .parse(input)
}

/// Parser for an eval tag
fn eval_parser(input: &str) -> IResult<&str, String> {
    map(
        (tag("eval"), spacing, alt((signed_number, mate_eval))),
        |(_, _, value)| format!("[eval {}]", value),
    )
    .parse(input)
}

/// Parser for a clk tag
fn clk_parser(input: &str) -> IResult<&str, String> {
    map((tag("clk"), spacing, time_value), |(_, _, value)| {
        format!("[clk {}]", value)
    })
    .parse(input)
}

/// Parser for a signed number
fn signed_number(input: &str) -> IResult<&str, String> {
    map(
        recognize(pair(
            opt(alt((char('+'), char('-')))),
            recognize(pair(digit1, opt(preceded(char('.'), digit1)))),
        )),
        |s: &str| s.to_string(),
    )
    .parse(input)
}

fn mate_eval(input: &str) -> IResult<&str, String> {
    let signed_integer = recognize((
        opt(char('-')), // Optional minus sign
        digit1,         // One or more digits
    ));
    map(preceded(char('#'), signed_integer), String::from).parse(input)
}

/// Parser for a time value
fn time_value(input: &str) -> IResult<&str, String> {
    map(
        (
            digit1,    // Hours
            char(':'), // Colon separator
            digit1,    // Minutes
            char(':'), // Colon separator
            digit1,    // Seconds
            opt(preceded(
                char('.'), // Dot separator
                digit1,    // Fractional seconds
            )),
        ),
        |(h, _, m, _, s, frac)| match frac {
            Some(f) => format!("{}:{}:{}.{}", h, m, s, f),
            None => format!("{}:{}:{}", h, m, s),
        },
    )
    .parse(input)
}

/// Parser for text (any characters except '[' and ']')
fn text(input: &str) -> IResult<&str, &str> {
    is_not("[]").parse(input)
}

/// Parser for spacing (one or more spaces)
fn spacing(input: &str) -> IResult<&str, &str> {
    recognize(many1(char(' '))).parse(input)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_comments1() {
        let input = "[%eval 123] some text [%clk 12:34:56]";
        let result = parse_comments(input);
        assert!(result.is_ok());
        let (remaining, parsed) = result.unwrap();
        assert_eq!(
            parsed,
            vec![
                CommentContent::Eval(123.0),
                CommentContent::Text(" some text ".to_string()),
                CommentContent::ClkTime((12, 34, 56.0))
            ]
        );
        assert_eq!(remaining, "");
    }

    #[test]
    fn test_comments2() {
        let input = "[%clk 12:34:56] some text ";
        let result = parse_comments(input);
        assert!(result.is_ok());
        let (remaining, parsed) = result.unwrap();
        assert_eq!(
            parsed,
            vec![
                CommentContent::ClkTime((12, 34, 56.0)),
                CommentContent::Text(" some text ".to_string())
            ]
        );
        assert_eq!(remaining, "");
    }

    #[test]
    fn test_comment_fractional_sec() {
        let input = "[%clk 12:34:56.0123]";
        let result = parse_comments(input);
        assert!(result.is_ok());
        let (remaining, parsed) = result.unwrap();
        assert_eq!(parsed, vec![CommentContent::ClkTime((12, 34, 56.0123)),]);
        assert_eq!(remaining, "");
    }

    #[test]
    fn test_tag_parser() {
        let input = "[%eval 123]";
        let result = tag_parser(input);
        assert!(result.is_ok());
        let (remaining, parsed) = result.unwrap();
        assert_eq!(parsed, "[eval 123]");
        assert_eq!(remaining, "");
    }

    #[test]
    fn test_eval_mate() {
        let input = "[%eval #-3]";
        let result = tag_parser(input);
        assert!(result.is_ok());
        let (remaining, parsed) = result.unwrap();
        assert_eq!(parsed, "[eval -3]"); // TODO mark the mates
        assert_eq!(remaining, "");
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
        let (remaining, parsed) = result.unwrap();
        assert_eq!(parsed, "[clk 12:34:56]");
        assert_eq!(remaining, "");
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
        let (remaining, parsed) = result.unwrap();
        assert_eq!(parsed, "some text");
        assert_eq!(remaining, "");
    }

    #[test]
    fn test_signed_number() {
        let input = "-123.45";
        let result = signed_number(input);
        assert!(result.is_ok());
        let (remaining, parsed) = result.unwrap();
        assert_eq!(parsed, "-123.45");
        assert_eq!(remaining, "");
    }

    #[test]
    fn test_time_value() {
        let input = "12:34:56";
        let result = time_value(input);
        assert!(result.is_ok());
        let (remaining, parsed) = result.unwrap();
        assert_eq!(parsed, "12:34:56");
        assert_eq!(remaining, "");
    }

    #[test]
    fn test_time_value_fractional() {
        let input = "12:34:56.12345";
        let result = time_value(input);
        assert!(result.is_ok());
        let (remaining, parsed) = result.unwrap();
        assert_eq!(parsed, "12:34:56.12345");
        assert_eq!(remaining, "");
    }
}
