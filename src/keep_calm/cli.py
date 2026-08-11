"""Keep Calm CLI tool."""

from __future__ import annotations

import sys

from keep_calm.analyzer import KeepCalmAnalyzer


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: keep-calm 'Your message here'")
        print("       echo 'message' | keep-calm")
        sys.exit(1)

    text = " ".join(sys.argv[1:]) if sys.argv[1:] else sys.stdin.read().strip()

    if not text:
        print("Error: no message provided")
        sys.exit(1)

    analyzer = KeepCalmAnalyzer()
    result = analyzer.analyze(text)

    print(f"\nRisk: {result.communication_risk:.2f} ({result.risk_level.value})")
    print(f"Tones: {', '.join(f'{t.label.value}({t.confidence:.2f})' for t in result.tones)}")
    print(f"Intent: {result.intent.value} ({result.intent_confidence:.2f})")
    print(f"Needs attention: {'yes' if result.needs_attention else 'no'}")
    print(f"\n{result.explanation}")


if __name__ == "__main__":
    main()
