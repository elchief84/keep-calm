# Keep Calm

> Think twice. Send once.

A lightweight, privacy-first AI model that helps people communicate better before sending a message.

Keep Calm analyzes the emotional and communication impact of a message and warns the user when a message could be perceived as aggressive, disrespectful, or harmful.

**Not a moderator. Not a censor. A communication companion.**

## Quick Start

```bash
pip install keep-calm
keep-calm "Your message here"
```

## What It Does

- Estimates communication risk (0-1 score + level)
- Detects emotional tone (5 categories)
- Classifies communication intent (4 categories)
- Provides specific, trigger-level explanations
- Confirms healthy communication
- Runs entirely locally — no data leaves your machine

## What It Does NOT Do

- Block or censor messages
- Rewrite or suggest rewrites
- Store or log messages
- Judge user intentions

## Languages

- English (MVP)
- Italian (MVP)

## Documentation

- [Architecture](ARCHITECTURE.md)
- [Annotation Guidelines](docs/annotation_guidelines.md)

## License

Apache 2.0
