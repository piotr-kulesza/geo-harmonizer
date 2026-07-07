# Mini design brief (read on Day 5 BEFORE any front-end code)

Goal: a striking front end that does NOT read as "AI-generated". Spend boldness
in one place (the PCA moment); keep everything else quiet and disciplined. This
is a starting direction — adapt, don't copy.

## Subject
A tool for a researcher drowning in mismatched data. World: expression matrices,
sample clouds, chaos → order. Draw the visual language from there.

## Avoid (AI defaults — judges will spot them at an Anthropic event)
- Cream background + serif display + terracotta **#D97757** (that's Claude's own
  accent — a double tell here). Do not use it.
- Black background + one neon accent (acid green / vermilion).
- Broadsheet: hairline rules + zero border-radius + dense columns.

## Tokens (propose your own; this is direction)
- **Palette:** a calm background (not cream, not default-black). Reserve saturated
  colour for **encoding dataset source in the PCA** — each dataset its own colour.
  Colour is functional there (structure carries information), not decoration.
- **Type:** a display + body pair that are NOT the Tailwind defaults, plus a
  utility/monospace face for numbers (gene counts, sample counts) — data is part
  of this product's identity.
- **Signature:** the PCA chaos→order transition is the ONLY bold thing.

## Motion
One orchestrated moment (the PCA) beats ten scattered effects. Excess animation
reads as AI-generated. Respect `prefers-reduced-motion`.

## Copy
Researcher's language, active verbs, sentence case. "Compare datasets" /
"Download unified data" — not "Run pipeline" / "Submit". Empty state = an
invitation to act. Error = what went wrong + how to fix it ("This series couldn't
be parsed automatically — upload the file manually"), not an apology.

## Quality floor (don't announce it)
Responsive to mobile, visible keyboard focus, reduced-motion. Before you finish:
look at the whole thing and remove one effect (take off one accessory before you
leave).
