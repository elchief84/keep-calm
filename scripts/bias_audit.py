"""Bias audit script for Keep Calm.

Runs the analyzer on curated test cases designed to probe for systematic bias
across communication styles, dialects, and cultural norms.

Usage:
    python scripts/bias_audit.py [--output data/results/bias_audit.json]
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from keep_calm.analyzer import KeepCalmAnalyzer


@dataclass
class BiasProbe:
    text: str
    category: str
    expected_max_level: str  # risk_level that would be acceptable
    note: str = ""


PROBES: list[BiasProbe] = [
    # ── Direct communication (DE/NL/IL style) ──
    BiasProbe("This is wrong. Fix it.", "direct_communication", "low",
              "Direct German/Dutch style — should not flag as hostile."),
    BiasProbe("I disagree with this approach entirely.", "direct_communication", "low",
              "Clear disagreement, not hostility."),
    BiasProbe("The design has several flaws.", "direct_communication", "low",
              "Direct critique of work, not person."),
    BiasProbe("We need to redo this from scratch.", "direct_communication", "low",
              "Direct but task-focused."),
    BiasProbe("This won't work. Here is why.", "direct_communication", "low",
              "Direct feedback with rationale."),
    BiasProbe("I need this by Friday.", "direct_communication", "none",
              "Direct deadline request — should be neutral."),
    BiasProbe("No, I don't think that is correct.", "direct_communication", "low",
              "Polite disagreement."),
    BiasProbe("This is not acceptable quality.", "direct_communication", "medium",
              "Blunt quality assessment."),

    # ── Indirect communication (UK/Japanese style) ──
    BiasProbe("I wonder if we might consider a different approach?", "indirect_communication", "none",
              "Very indirect suggestion."),
    BiasProbe("This is interesting, though I'm not entirely sure it aligns.", "indirect_communication", "none",
              "Soft disagreement."),
    BiasProbe("Perhaps we could revisit this when we have more data?", "indirect_communication", "none",
              "Deferred polite pushback."),
    BiasProbe("With respect, I have some reservations about this direction.", "indirect_communication", "low",
              "Polite disagreement with deference markers."),
    BiasProbe("I'm not sure I fully understand the reasoning behind this decision.", "indirect_communication", "low",
              "Indirect challenge."),

    # ── AAVE / dialect ──
    BiasProbe("This feature is fire, great work.", "aave", "none",
              "AAVE positive expression — should not be penalized."),
    BiasProbe("No cap, this needs more work.", "aave", "low",
              "AAVE honest feedback."),
    BiasProbe("I ain't saying it's bad, just needs some polish.", "aave", "low",
              "AAVE constructive feedback."),
    BiasProbe("Y'all did amazing on this release.", "aave", "none",
              "AAVE positive message."),
    BiasProbe("We been working on this for weeks and it keeps breaking.", "aave", "medium",
              "AAVE frustrated but valid."),

    # ── Non-native English ──
    BiasProbe("You no understand this problem.", "non_native_english", "medium",
              "NNS phrasing — intent is critical, not necessarily hostile."),
    BiasProbe("Please to explain the code. I not understanding.", "non_native_english", "none",
              "NNS asking for help."),
    BiasProbe("The documentation is not good. We need better.", "non_native_english", "low",
              "NNS blunt but constructive."),
    BiasProbe("I am sorry, but I think you mistaken on this point.", "non_native_english", "low",
              "NNS polite disagreement."),
    BiasProbe("Why this always happening? Very frustrating.", "non_native_english", "medium",
              "NNS expressing frustration about situation."),

    # ── Neurodivergent communication ──
    BiasProbe("Here are the exact steps as required:\n1. Run the build\n2. Check output\n3. Deploy",
              "neurodivergent", "none", "Precise, structured communication."),
    BiasProbe("I need specific details about what went wrong, not a summary.",
              "neurodivergent", "low", "Request for details, not aggressive."),
    BiasProbe("This is factually incorrect per the documentation on line 45.",
              "neurodivergent", "low", "Factual correction with evidence."),
    BiasProbe("Can you provide the exact error message and stack trace?",
              "neurodivergent", "none", "Direct technical request."),
    BiasProbe("Your explanation does not answer the question I asked. I asked about X, you answered about Y.",
              "neurodivergent", "low", "Precise frustration about miscommunication."),
    BiasProbe("I do not understand the feedback. Please be more specific.",
              "neurodivergent", "none", "Clear request for clarification."),

    # ── Italian directness ──
    BiasProbe("Questo approccio ha diversi problemi.", "italian_direct", "low",
              "IT: Direct critique of approach — normal in Italian workplace."),
    BiasProbe("Non sono d'accordo, spiego perché.", "italian_direct", "low",
              "IT: Clear disagreement with rationale."),
    BiasProbe("Ma che stai dicendo? Non ha senso.", "italian_direct", "medium",
              "IT: Confrontational phrasing — medium risk expected."),
    BiasProbe("Ottimo lavoro, sono molto contento del risultato!", "italian_direct", "none",
              "IT: Enthusiastic praise — should be positive."),
    BiasProbe("Guarda che così non funziona, te lo dico subito.", "italian_direct", "medium",
              "IT: Blunt but constructive warning."),

    # ── Short ambiguous messages ──
    BiasProbe("fine.", "ambiguous_short", "medium", "Sarcastic/dismissive in workplace."),
    BiasProbe("ok", "ambiguous_short", "none", "Neutral acknowledgment."),
    BiasProbe("k", "ambiguous_short", "low", "Very terse — could be dismissive."),
    BiasProbe("lol ok", "ambiguous_short", "low", "Informal acknowledgment."),
    BiasProbe("whatever", "ambiguous_short", "medium", "Dismissive."),
    BiasProbe("...", "ambiguous_short", "medium", "Ellipsis — highly ambiguous."),
    BiasProbe("sure", "ambiguous_short", "none", "Agreement."),
    BiasProbe("interesting", "ambiguous_short", "low", "Can be genuine or sarcastic."),

    # ── Constructive low risk (control group) ──
    BiasProbe("Thanks for the update, this looks great!", "control_benign", "none", ""),
    BiasProbe("Let me know if you need any help with this.", "control_benign", "none", ""),
    BiasProbe("Good point, I'll incorporate that feedback.", "control_benign", "none", ""),
    BiasProbe("Great work on this PR — the tests are thorough.", "control_benign", "none", ""),
    BiasProbe("I appreciate the effort you put into this analysis.", "control_benign", "none", ""),

    # ── Clearly toxic (control group) ──
    BiasProbe("You are completely incompetent and should be fired.", "control_toxic", "critical", ""),
    BiasProbe("Nobody here respects your opinion.", "control_toxic", "high", ""),
    BiasProbe("Shut up and let the adults talk.", "control_toxic", "critical", ""),
    BiasProbe("You are the worst developer I've ever worked with.", "control_toxic", "critical", ""),
]


def run_audit(analyzer: KeepCalmAnalyzer) -> list[dict[str, Any]]:
    results = []
    for i, probe in enumerate(PROBES):
        result = analyzer.analyze(probe.text)
        level_order = {"none": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
        expected_max_score = level_order[probe.expected_max_level]
        actual_score = level_order[result.risk_level.value]
        flagged_too_high = actual_score > expected_max_score

        results.append({
            "id": i,
            "text": probe.text,
            "category": probe.category,
            "note": probe.note,
            "expected_max_level": probe.expected_max_level,
            "risk_score": result.communication_risk,
            "risk_level": result.risk_level.value,
            "tones": [{"label": t.label.value, "confidence": t.confidence} for t in result.tones],
            "intent": result.intent.value,
            "needs_attention": result.needs_attention,
            "flagged_too_high": flagged_too_high,
            "explanation": result.explanation,
            "severity_gap": actual_score - expected_max_score,
        })
    return results


def report(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_category: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        by_category.setdefault(r["category"], []).append(r)

    summary: dict[str, Any] = {}
    for cat, items in sorted(by_category.items()):
        false_positives = [i for i in items if i["flagged_too_high"]]
        avg_risk = sum(i["risk_score"] for i in items) / len(items)
        avg_gap = sum(i["severity_gap"] for i in items) / len(items)
        summary[cat] = {
            "count": len(items),
            "false_positives": len(false_positives),
            "fp_rate": round(len(false_positives) / len(items), 3),
            "avg_risk_score": round(avg_risk, 3),
            "avg_severity_gap": round(avg_gap, 3),
        }
        if false_positives:
            summary[cat]["fp_examples"] = [
                {"text": fp["text"], "level": fp["risk_level"], "expected_max": fp["expected_max_level"]}
                for fp in false_positives
            ]

    total_fp = sum(1 for r in results if r["flagged_too_high"])
    return {
        "total_probes": len(results),
        "total_false_positives": total_fp,
        "overall_fp_rate": round(total_fp / len(results), 3),
        "by_category": summary,
        "details": results,
    }


def main() -> None:
    output_path = None
    args = sys.argv[1:]
    if len(args) >= 2 and args[0] == "--output":
        output_path = Path(args[1])

    print("Loading analyzer and running bias audit...")
    analyzer = KeepCalmAnalyzer()
    results = run_audit(analyzer)
    report_data = report(results)

    print(f"\n{'='*60}")
    print("BIAS AUDIT REPORT")
    print(f"{'='*60}")
    print(f"Total probes: {report_data['total_probes']}")
    print(f"False positives (flagged too high): {report_data['total_false_positives']}")
    print(f"Overall FP rate: {report_data['overall_fp_rate']:.1%}")
    print()

    for cat, stats in sorted(report_data["by_category"].items()):
        indicator = "WARN" if stats["fp_rate"] > 0.3 else " OK "
        print(f"  [{indicator}] {cat}:")
        print(f"         {stats['count']} probes, {stats['false_positives']} FPs "
              f"({stats['fp_rate']:.0%}), avg risk {stats['avg_risk_score']}, avg gap {stats['avg_severity_gap']}")
        if "fp_examples" in stats:
            for ex in stats["fp_examples"]:
                print(f"         -> \"{ex['text'][:70]}\"  [{ex['level']} > expected {ex['expected_max']}]")
    print()

    # Specific checks
    direct_cat = report_data["by_category"].get("direct_communication", {})
    indirect_cat = report_data["by_category"].get("indirect_communication", {})
    if direct_cat and indirect_cat:
        gap = direct_cat["avg_risk_score"] - indirect_cat["avg_risk_score"]
        print(f"Direct vs Indirect communication gap: {gap:+.3f} risk score")
        if gap > 0.15:
            print("WARNING: Direct communication is systematically penalized relative to indirect.")
        print()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report_data, indent=2))
        print(f"Full report saved to {output_path}")


if __name__ == "__main__":
    main()
