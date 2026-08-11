# References

Provenance for methodological decisions: which published work a choice in this codebase rests
on, what that work actually says, and where in the code the choice lives. A reviewer should be
able to start from any cited line and reach the sentence in the paper that justifies it.

`python scripts/verify_citations.py` checks all of this mechanically. Run it before trusting
anything here.

## What gets a citation

Methods, not values. This platform derives parameters from the data at runtime rather than
freezing them, so a citation attached to a literal constant is a smell rather than provenance: it
means something was pinned that should have been derived. Where a number does appear, the
citation justifies the derivation procedure, never the number.

Cite a decision when a reviewer could reasonably ask "why this and not something else" and the
answer is methodological. Metric choice, split strategy, loss selection under class imbalance,
augmentation policy, milestone definition, calibration approach, agreement statistics. Not
implementation detail, and not choices that are matters of engineering taste.

## Layout

One file per publication, `references/<key>.md`, where `<key>` matches the filename stem and is
`firstauthor` plus year plus a short word, lowercase, hyphenated. PDFs live in
`references/pdf/<key>.pdf` and are not tracked, because redistributing paywalled papers is
infringement and the files are large. The metadata, quotes, and anchors are tracked, so CI can
check everything except the quote match, and the quote match runs for anyone who has fetched the
papers. Record `url` so fetching is reproducible.

## Format

See `_template.md`. Frontmatter carries the bibliographic record and a `supports` list; the body
carries prose about what the paper does and does not establish.

Anchors are `path::symbol`, never line numbers. Line numbers are wrong within a week, and a
rotted anchor cannot be told apart from a wrong one. Nested symbols use `path::Class.method`.

Quotes are verbatim. The verifier normalizes whitespace and the hyphenation that PDF extraction
introduces, then requires the quote to appear in the extracted text. A paraphrase will fail, which
is the point: it is the one check an invented quote cannot survive.

## Marking the code

Put a single line at the decision site:

    # cite: hosang2017-nms

The verifier requires every marker to resolve to a reference and every reference to be reached by
at least one marker, so the link cannot rot in either direction. These are literature citations
recording a standing reason a decision is what it is. They are not the session or project
tracking citations the repository prose rules forbid, and should not be removed as such.

## Adding an entry

The bar is higher for an agent than for a person. An agent may add a reference only with a
resolvable DOI or URL, a stored PDF, and a quote that passes the verbatim check. If it cannot
obtain the paper it records the gap as an open question rather than citing from memory. Citing
from memory is where fabricated references enter, and a fabricated citation in a scientific
platform is worse than no citation at all, because it manufactures authority that a reviewer will
believe.
