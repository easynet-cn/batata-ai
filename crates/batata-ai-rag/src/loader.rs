//! Document loaders: turn raw files into plain text ready for chunking.
//!
//! Loaders are intentionally simple free functions instead of full
//! `DocumentLoader` trait objects — the trait lives in `batata-ai-core`
//! for future non-trivial cases (remote URLs, lazy streams), but the
//! ones shipped here are CPU-only, synchronous, and do not need dynamic
//! dispatch.
//!
//! All parsers are pure Rust. No system dependencies (no poppler, no
//! libxml2, etc.) — the point of `batata-ai` is a self-contained build.

use std::path::Path;

use batata_ai_core::error::{BatataError, Result};

/// Supported document formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DocumentFormat {
    PlainText,
    Markdown,
    Html,
    Pdf,
}

impl DocumentFormat {
    /// Guess the format from a MIME type, falling back to the file extension.
    pub fn detect(mime: Option<&str>, path: &Path) -> Self {
        if let Some(mime) = mime {
            let mime = mime.to_ascii_lowercase();
            match mime.as_str() {
                "application/pdf" => return Self::Pdf,
                "text/html" | "application/xhtml+xml" => return Self::Html,
                "text/markdown" | "text/x-markdown" => return Self::Markdown,
                "text/plain" => return Self::PlainText,
                _ => {}
            }
        }
        match path
            .extension()
            .and_then(|e| e.to_str())
            .map(|s| s.to_ascii_lowercase())
            .as_deref()
        {
            Some("pdf") => Self::Pdf,
            Some("html" | "htm" | "xhtml") => Self::Html,
            Some("md" | "markdown") => Self::Markdown,
            _ => Self::PlainText,
        }
    }

    /// The canonical MIME type for this format.
    pub fn mime(&self) -> &'static str {
        match self {
            Self::PlainText => "text/plain",
            Self::Markdown => "text/markdown",
            Self::Html => "text/html",
            Self::Pdf => "application/pdf",
        }
    }
}

/// Decode plain-text bytes (lossy UTF-8).
fn plain_from_bytes(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

/// Load plain text from a file.
fn load_plain(path: &Path) -> Result<String> {
    std::fs::read_to_string(path).map_err(BatataError::from)
}

/// Load a Markdown file, stripping syntax so the embedder sees content
/// rather than structural tokens.
fn load_markdown(path: &Path) -> Result<String> {
    let raw = std::fs::read_to_string(path).map_err(BatataError::from)?;
    Ok(markdown_to_plain(&raw))
}

/// Render markdown to plain text via pulldown-cmark.
pub fn markdown_to_plain(raw: &str) -> String {
    use pulldown_cmark::{Event, Parser, Tag, TagEnd};

    let parser = Parser::new(raw);
    let mut out = String::with_capacity(raw.len());
    let mut need_newline = false;

    for event in parser {
        match event {
            Event::Text(t) | Event::Code(t) => {
                out.push_str(&t);
                need_newline = true;
            }
            Event::SoftBreak => out.push(' '),
            Event::HardBreak => out.push('\n'),
            Event::End(TagEnd::Paragraph | TagEnd::Heading(_) | TagEnd::Item) => {
                if need_newline {
                    out.push('\n');
                    need_newline = false;
                }
            }
            Event::Start(Tag::CodeBlock(_)) | Event::End(TagEnd::CodeBlock) => {
                if need_newline {
                    out.push('\n');
                    need_newline = false;
                }
            }
            _ => {}
        }
    }
    out.trim().to_string()
}

/// Load an HTML file and flatten to readable plain text.
fn load_html(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path).map_err(BatataError::from)?;
    html_bytes_to_plain(&bytes)
}

/// Convert HTML bytes to plain text using html2text. Widening to ~120
/// columns keeps wrapping sane without introducing hard breaks inside
/// sentences.
pub fn html_bytes_to_plain(bytes: &[u8]) -> Result<String> {
    html2text::from_read(bytes, 120).map_err(|e| {
        BatataError::Inference(format!("html2text failed: {e}"))
    })
}

/// Extract text from a PDF file.
fn load_pdf(path: &Path) -> Result<String> {
    // pdf-extract panics on some malformed PDFs; catch it so a bad file
    // fails the ingest cleanly instead of taking down the worker.
    let path = path.to_path_buf();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        pdf_extract::extract_text(&path)
    }));
    match result {
        Ok(Ok(text)) => Ok(text),
        Ok(Err(e)) => Err(BatataError::Inference(format!("pdf-extract failed: {e}"))),
        Err(_) => Err(BatataError::Inference(
            "pdf-extract panicked while parsing PDF".to_string(),
        )),
    }
}

/// Extract text from in-memory PDF bytes.
fn pdf_from_bytes(bytes: &[u8]) -> Result<String> {
    let bytes = bytes.to_vec();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        pdf_extract::extract_text_from_mem(&bytes)
    }));
    match result {
        Ok(Ok(text)) => Ok(text),
        Ok(Err(e)) => Err(BatataError::Inference(format!("pdf-extract failed: {e}"))),
        Err(_) => Err(BatataError::Inference(
            "pdf-extract panicked while parsing PDF".to_string(),
        )),
    }
}

/// Decode in-memory bytes into text according to `format`.
pub fn load_bytes(bytes: &[u8], format: DocumentFormat) -> Result<String> {
    match format {
        DocumentFormat::PlainText => Ok(plain_from_bytes(bytes)),
        DocumentFormat::Markdown => Ok(markdown_to_plain(&plain_from_bytes(bytes))),
        DocumentFormat::Html => html_bytes_to_plain(bytes),
        DocumentFormat::Pdf => pdf_from_bytes(bytes),
    }
}

/// Load any supported file type. `mime` is an optional hint; when absent
/// the format is inferred from the file extension.
pub fn load_file(path: &Path, mime: Option<&str>) -> Result<(DocumentFormat, String)> {
    let format = DocumentFormat::detect(mime, path);
    let text = match format {
        DocumentFormat::PlainText => load_plain(path)?,
        DocumentFormat::Markdown => load_markdown(path)?,
        DocumentFormat::Html => load_html(path)?,
        DocumentFormat::Pdf => load_pdf(path)?,
    };
    Ok((format, text))
}

/// Load from a URI that may be `file://` or a plain filesystem path.
pub fn load_uri(source_uri: &str, mime: Option<&str>) -> Result<(DocumentFormat, String)> {
    let path_str = source_uri
        .strip_prefix("file://")
        .unwrap_or(source_uri);
    let path = Path::new(path_str);
    if !path.exists() {
        return Err(BatataError::NotFound(format!(
            "loader source file not found: {}",
            path.display()
        )));
    }
    load_file(path, mime)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn tmp_with(name: &str, content: &[u8]) -> std::path::PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!("batata-ai-loader-{}-{}", std::process::id(), name));
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(content).unwrap();
        path
    }

    #[test]
    fn detect_by_extension() {
        let p = Path::new("/tmp/x.pdf");
        assert_eq!(DocumentFormat::detect(None, p), DocumentFormat::Pdf);
        let p = Path::new("/tmp/x.md");
        assert_eq!(DocumentFormat::detect(None, p), DocumentFormat::Markdown);
        let p = Path::new("/tmp/x.html");
        assert_eq!(DocumentFormat::detect(None, p), DocumentFormat::Html);
        let p = Path::new("/tmp/x");
        assert_eq!(DocumentFormat::detect(None, p), DocumentFormat::PlainText);
    }

    #[test]
    fn detect_by_mime_beats_extension() {
        let p = Path::new("/tmp/x.txt");
        assert_eq!(
            DocumentFormat::detect(Some("application/pdf"), p),
            DocumentFormat::Pdf
        );
    }

    #[test]
    fn markdown_to_plain_strips_syntax() {
        let md = "# Title\n\nSome **bold** and *italic* [link](http://x) text.\n\n- item 1\n- item 2\n";
        let out = markdown_to_plain(md);
        assert!(out.contains("Title"));
        assert!(out.contains("bold"));
        assert!(out.contains("italic"));
        assert!(out.contains("item 1"));
        assert!(!out.contains("**"));
        assert!(!out.contains("["));
    }

    #[test]
    fn html_to_plain_strips_tags() {
        let html = b"<html><body><h1>Title</h1><p>Hello nice day.</p></body></html>";
        let out = html_bytes_to_plain(html).unwrap();
        assert!(out.to_lowercase().contains("title"));
        assert!(out.to_lowercase().contains("hello nice day"));
        // Raw HTML tags must be gone.
        assert!(!out.contains("<h1>"));
        assert!(!out.contains("</body>"));
    }

    #[test]
    fn load_file_plain_text() {
        let p = tmp_with("plain.txt", b"line1\nline2\n");
        let (fmt, text) = load_file(&p, None).unwrap();
        assert_eq!(fmt, DocumentFormat::PlainText);
        assert!(text.contains("line1"));
        let _ = std::fs::remove_file(p);
    }

    #[test]
    fn load_file_markdown() {
        let p = tmp_with(
            "doc.md",
            b"# Header\n\nSome **strong** content.\n\n- a\n- b\n",
        );
        let (fmt, text) = load_file(&p, None).unwrap();
        assert_eq!(fmt, DocumentFormat::Markdown);
        assert!(text.contains("Header"));
        assert!(!text.contains("**"));
        let _ = std::fs::remove_file(p);
    }

    #[test]
    fn load_file_html() {
        let p = tmp_with("doc.html", b"<html><body><p>Plain text.</p></body></html>");
        let (fmt, text) = load_file(&p, None).unwrap();
        assert_eq!(fmt, DocumentFormat::Html);
        assert!(text.to_lowercase().contains("plain text"));
        let _ = std::fs::remove_file(p);
    }

    #[test]
    fn load_uri_rejects_missing() {
        let err = load_uri("file:///nonexistent/abcdef.txt", None).unwrap_err();
        assert!(matches!(err, BatataError::NotFound(_)));
    }
}
