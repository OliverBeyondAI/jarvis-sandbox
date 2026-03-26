# Skill: Code Tasks

General-purpose coding skill for creating, modifying, and enhancing files (HTML, CSS, JS, etc.). Covers landing pages, UI components, feature additions, and file modifications.

---

## Step-by-Step Checklist

### 1. Understand the Request
- [ ] Read the task requirements literally — identify every specific deliverable and constraint.
- [ ] List the primary output files by name (e.g., `index.html`).
- [ ] Note any explicit constraints: "no dependencies," "keep existing design," "single file," etc.

### 2. Read Before Writing
- [ ] If modifying an existing file, read the entire file first to understand structure, style, and design language.
- [ ] Identify where new elements belong based on the actual page structure — never assume.
- [ ] Note existing patterns: CSS approach (variables, inline, classes), spacing, color palette, typography.

### 3. Implement with Minimal Impact
- [ ] Create the primary deliverable file(s) **first**, before any supporting files.
- [ ] For modifications: make targeted edits — do not rewrite or restructure code beyond what the task requires.
- [ ] Match new elements to the existing design language (colors, fonts, spacing, component patterns).
- [ ] Prefer inline/self-contained solutions (inline SVG, data URIs, CSS variables) over external dependencies.
- [ ] Include mobile-responsive design: viewport meta tag and media queries.

### 4. Verify the Output
- [ ] Re-read the task requirements and confirm every item was addressed.
- [ ] Read back every output file to confirm it exists on disk and contains the expected content.
- [ ] Check that HTML is well-formed: all tags properly closed, no broken nesting.
- [ ] Confirm zero external dependencies if the task specifies self-contained output.

---

## Quality Requirements — Definition of Done

- **Primary deliverable exists and is complete.** The file(s) named in the task are present, readable, and contain all requested content.
- **All requested features are present.** Every section, element, or behavior explicitly asked for is implemented.
- **Design is clean and consistent.** New code matches existing style. CSS variables used for theming. Good whitespace and typography.
- **Mobile-responsive.** Pages render correctly on small screens with appropriate breakpoints.
- **Self-contained when specified.** No CDN links, external fonts, or JS frameworks unless the task explicitly allows them.
- **Existing content preserved.** When modifying files, all prior content and styling remains intact unless removal was requested.

---

## Common Mistakes to Avoid

### Critical — These cause task failure
- **Not creating the deliverable file.** Committing config/meta files without the actual requested output is a zero-score result. Always create the primary artifact first.
- **Not verifying output exists.** Never assume a file was written — read it back to confirm.
- **Skipping requirements.** If the task says "include a CTA button," there must be a CTA button. Read requirements literally.

### Design & Implementation
- **Rewriting existing code unnecessarily.** When the task says "add X," only add X. Don't restructure, refactor, or restyle what's already there.
- **Assuming page structure.** Always read the file first to know where new elements should go.
- **Adding external dependencies.** Don't pull in CDN links, Google Fonts, or JS libraries when inline solutions work.
- **Over-engineering.** A simple landing page doesn't need JS animations, build tools, or component frameworks. Minimal and modern wins.
- **Forgetting mobile.** Always include `<meta name="viewport">` and at least one media query breakpoint.

---

## Oliver's Preferences

- **Single-file, zero-dependency output** is the default for HTML/CSS tasks unless told otherwise.
- **CSS variables** for colors and spacing — keeps theming consistent and maintainable.
- **Minimal, modern aesthetic** with good whitespace and typography over heavy decoration.
- **Hero section pattern** (headline + description + CTA) is the standard landing page structure.
- **Inline SVG / data URIs** preferred for icons and favicons — no image file dependencies.
- **Read-first workflow** — always understand what exists before touching it.
- **Verify-last workflow** — always confirm the output is real and complete before declaring done.
