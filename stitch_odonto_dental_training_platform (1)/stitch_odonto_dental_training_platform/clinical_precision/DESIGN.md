---
name: Clinical Precision
colors:
  surface: '#f8f9ff'
  surface-dim: '#cbdbf5'
  surface-bright: '#f8f9ff'
  surface-container-lowest: '#ffffff'
  surface-container-low: '#eff4ff'
  surface-container: '#e5eeff'
  surface-container-high: '#dce9ff'
  surface-container-highest: '#d3e4fe'
  on-surface: '#0b1c30'
  on-surface-variant: '#42474f'
  inverse-surface: '#213145'
  inverse-on-surface: '#eaf1ff'
  outline: '#72777f'
  outline-variant: '#c2c7d0'
  surface-tint: '#35618d'
  primary: '#00375e'
  on-primary: '#ffffff'
  primary-container: '#1f4e79'
  on-primary-container: '#95bff1'
  inverse-primary: '#a0cafc'
  secondary: '#006a61'
  on-secondary: '#ffffff'
  secondary-container: '#86f2e4'
  on-secondary-container: '#006f66'
  tertiary: '#542a00'
  on-tertiary: '#ffffff'
  tertiary-container: '#753d00'
  on-tertiary-container: '#ffa85e'
  error: '#ba1a1a'
  on-error: '#ffffff'
  error-container: '#ffdad6'
  on-error-container: '#93000a'
  primary-fixed: '#d1e4ff'
  primary-fixed-dim: '#a0cafc'
  on-primary-fixed: '#001d35'
  on-primary-fixed-variant: '#184974'
  secondary-fixed: '#89f5e7'
  secondary-fixed-dim: '#6bd8cb'
  on-secondary-fixed: '#00201d'
  on-secondary-fixed-variant: '#005049'
  tertiary-fixed: '#ffdcc3'
  tertiary-fixed-dim: '#ffb77d'
  on-tertiary-fixed: '#2f1500'
  on-tertiary-fixed-variant: '#6e3900'
  background: '#f8f9ff'
  on-background: '#0b1c30'
  surface-variant: '#d3e4fe'
typography:
  display-lg:
    fontFamily: Inter
    fontSize: 36px
    fontWeight: '600'
    lineHeight: 44px
    letterSpacing: -0.02em
  display-lg-mobile:
    fontFamily: Inter
    fontSize: 28px
    fontWeight: '600'
    lineHeight: 36px
    letterSpacing: -0.02em
  headline-lg:
    fontFamily: Inter
    fontSize: 28px
    fontWeight: '600'
    lineHeight: 36px
    letterSpacing: -0.015em
  headline-lg-mobile:
    fontFamily: Inter
    fontSize: 22px
    fontWeight: '600'
    lineHeight: 28px
    letterSpacing: -0.015em
  headline-md:
    fontFamily: Inter
    fontSize: 20px
    fontWeight: '600'
    lineHeight: 28px
    letterSpacing: -0.01em
  headline-sm:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '600'
    lineHeight: 24px
    letterSpacing: -0.005em
  body-lg:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 24px
    letterSpacing: 0em
  body-md:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 20px
    letterSpacing: 0em
  body-sm:
    fontFamily: Inter
    fontSize: 12px
    fontWeight: '400'
    lineHeight: 16px
    letterSpacing: 0.005em
  label-lg:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '500'
    lineHeight: 20px
    letterSpacing: 0.01em
  label-md:
    fontFamily: Inter
    fontSize: 12px
    fontWeight: '500'
    lineHeight: 16px
    letterSpacing: 0.01em
  label-sm:
    fontFamily: Inter
    fontSize: 11px
    fontWeight: '600'
    lineHeight: 14px
    letterSpacing: 0.04em
  code-md:
    fontFamily: JetBrains Mono
    fontSize: 12px
    fontWeight: '500'
    lineHeight: 16px
    letterSpacing: 0em
rounded:
  sm: 0.25rem
  DEFAULT: 0.5rem
  md: 0.75rem
  lg: 1rem
  xl: 1.5rem
  full: 9999px
spacing:
  gutter: 1.25rem
  gutter-mobile: 0.75rem
  margin: 2rem
  margin-mobile: 1rem
  space-xs: 0.25rem
  space-sm: 0.5rem
  space-md: 1rem
  space-lg: 1.5rem
  space-xl: 2.5rem
---

## Brand & Style

This design system delivers an exacting, institutional, and reassuring environment built for dental simulation, radiographic diagnostics, and clinical education. It merges the surgical discipline of medical software with the fluid ergonomics of modern browser applications. 

The aesthetic is grounded in **Clinical Modernism**:
- **High-legibility diagnostic density:** Information is surfaced with intentional restraint, ensuring high signal-to-noise ratios during high-stress case simulations.
- **Calm authority:** Generous clinical whitespace balanced by structured boundary lines eliminates cognitive clutter.
- **Instrumental clarity:** Interactive viewers, FDI tooth notation grids, and annotation tools feel like surgical instruments—calibrated, responsive, and predictable.

The design targets dental students, residents, and licensed practitioners. The interface avoids frivolous ornamentation, establishing clinical trust through crisp contrast, structural alignment, and disciplined color attribution.

## Colors

The palette establishes strict semantic accountability: color is never decorative; it is diagnostic.

### Primary & Brand Anchors
- **Primary Navy (`#1F4E79`)**: Conveys clinical authority, institutional trust, and structure. Used for active navigation states, primary operational triggers, patient header bands, and structural emphasis.
- **Deep Navy Shade (`#163756`)**: Hover states and authoritative headings.
- **Navy Tint (`#EBF2F8`)**: Interactive selection states and active row highlights.

### Clinical & Diagnostic Semantics
- **Clinical Teal (`#0D9488`)**: Denotes clinical verification, healthy margins, resolved simulations, and success indicators. Soft variant (`#14B8A6`) is reserved for high-contrast dark viewers.
- **Diagnostic Amber (`#D97706`)**: Reserved exclusively for AI predictions, diagnostic alerts, machine learning confidence scores, and unconfirmed anomalies. Soft amber tint (`#FEF3C7`) forms the background of prediction callouts.
- **Marker Red (`#EF4444`)**: Critical anomalies, structural bone loss warnings, pulpal involvement markers, and cancellation actions.
- **Clinical Violet (`#7C3AED`)**: Specialized auxiliary state for non-dental anatomical structures or cross-discipline dental specializations (e.g., ortho tracing).

### Neutrals & Surfaces
- **Canvas (`#F8FAFC`)**: Soft cool off-white that prevents clinical eye fatigue during extended radiograph review.
- **Sub-canvas / Well (`#F1F5F9`)**: Workstation wells, viewer frames, and inactive data toolbars.
- **Border Surface (`#E2E8F0`)**: Crisp, low-contrast structural boundaries.
- **Muted Slate (`#64748B`)**: Secondary metadata, measurement units, and reference labels.
- **Deep Slate (`#334155`)**: Form labels, table headings, and secondary actions.
- **Obsidian Slate (`#0F172A`)**: Primary patient data, diagnostic values, and core typography.

## Typography

The type system runs entirely on **Inter**, configured for optimal clinical readability, with **JetBrains Mono** deployed selectively for raw coordinate outputs, radiographic pixel dimensions, and tooth numerical indexes.

### Diagnostic Typography Rules
- **Tabular Figures (`tnum`)**: All clinical tables, periodontal depth matrices, millimeter measurements, and age/record fields must enforce OpenType tabular figures (`font-variant-numeric: tabular-nums`) to preserve optical column alignment.
- **FDI Notation**: Universal dental numbering and FDI notations (e.g., `#18`, `#21`, `d. 36`) are formatted using `label-md` with `font-feature-settings: "cv05", "cv11"` to prevent mistaking '1' for capital 'I' or '7'.
- **Hierarchical Discipline**: Section heads in simulation trays leverage `headline-sm` in all caps only when designating anatomical quadrants (e.g., `MAXILLARY RIGHT — Q1`), set with positive letter-spacing (`0.06em`).

## Layout & Spacing

The layout is engineered as a responsive, desktop-first clinical workspace utilizing a multi-pane layout model.

### Grid & Canvas Structure
- **Desktop Grid (>= 1280px)**: 12-column adaptive layout with a locked three-panel operational mode during active simulation:
  - **Left Rail (320px fixed)**: Patient chart, medical history, tooth selector.
  - **Center Stage (Fluid, min 640px)**: Radiographic/3D STL canvas and simulation staging.
  - **Right Inspector (360px fixed)**: Diagnostic findings, pathology inputs, and AI differential predictions.
- **Tablet Grid (768px - 1279px)**: Inspector and Patient History collapse into slide-over panels or tabbed bottom sheets, giving priority to the radiographic viewer.
- **Mobile Grid (< 768px)**: Single column with persistent bottom action bars for case submissions and floating toolbar controls for tooth quadrant traversal.

### Spatial Rhythm
- **Data Densities**: Clinical dashboards utilize compact `space-xs` (4px) and `space-sm` (8px) gaps between related diagnostic tags, measurements, and table cells.
- **Canvas Separations**: Outer cards and distinct anatomical inspection blocks breathe with `space-lg` (24px) internal padding and `space-xl` (40px) section breaks.

## Elevation & Depth

This system avoids expressive or dramatic dropshadows. Depth is functional, evoking clean surgical surfaces and backlit instrumentation.

### Layer Hierarchy
1. **Base Layer (`#F8FAFC`)**: Canvas level. The non-interactive operating background.
2. **Surface Layer (`#FFFFFF`)**: Patient cards, inspection modules, chart panels. Bound with a subtle structural border (`1px solid #E2E8F0`).
3. **Recessed Wells (`#F1F5F9`)**: Container for dental viewers, interactive panoramic x-rays (OPGs), and bite-wing frames. Recessed using `inset 0 1px 2px rgba(15, 23, 42, 0.06)`.
4. **Active Workspace Floating Elements**: Floating annotation palettes, measurement callouts, and tooth locator badges use an ambient, low-spread shadow: `0 4px 12px -2px rgba(31, 78, 121, 0.08), 0 2px 4px -1px rgba(15, 23, 42, 0.04)`.
5. **Modal & Critical Interrupters**: Focused diagnostic comparisons and simulation end-state summaries: `0 20px 25px -5px rgba(15, 23, 42, 0.12), 0 8px 10px -6px rgba(15, 23, 42, 0.06)` with a backdrop blur `backdrop-filter: blur(4px)` over an `#0F172A` tint at 40% opacity.

## Shapes

The geometric framework balances humanized approachability with clinical precision:

- **Cards & Primary Modules**: Strict 8px (`rounded-md`) to 12px (`rounded-lg`) curvature. Prevents the application from feeling abrasive while retaining geometric discipline.
- **Input Fields & Selectors**: 8px (`rounded-md`) corners to align flush with card layout structures.
- **Anatomical Badges & Notations**: Fully rounded pill shapes (`9999px`) are strictly reserved for FDI tooth indicators, diagnostic status markers, and user-assigned tags to contrast immediately against rectangular data modules.
- **Interactive Dental Viewer Canvas**: 12px (`rounded-lg`) border radius with an internal clip-path to ensure viewport overlays (rulers, contrast sliders) terminate cleanly at the perimeter.

## Components

### Buttons
- **Primary Clinical Action**: Deep Navy background (`#1F4E79`), pure white label, 8px border radius, 0 1px 2px rgba(0,0,0,0.05) shadow. Hover: `#163756`. Active: `#10273D`.
- **Secondary / Action Tool**: Clean white background, 1px border in `#E2E8F0`, Deep Slate text (`#334155`). Hover: `#F8FAFC` with border `#CBD5E1`.
- **AI Assist / Suggestion**: Amber-tinted background (`#FEF3C7`), border in `rgba(217, 119, 6, 0.3)`, text `#B45309`. Leading icon is an algorithmic sparkle or neural indicator.
- **Destructive Marker Action**: Danger Red (`#EF4444`) fill or 1px border with `#FEE2E2` hover state for clearing marked pathologies or re-setting case paths.

### Badges & Specialized Notations
- **FDI Tooth Badge**: Compact 24x24px or 28x28px circular/pill container. Dark Navy border (`#1F4E79`), pale blue-grey fill (`#F1F5F9`), bold monospace typography displaying numbers like `16`, `21`, or `48`. Active state swaps to filled `#1F4E79` with white text.
- **AI Prediction Chip**: `#FFFBEB` background, `#D97706` text, micro-badge displaying confidence tier (`AI 94%`). Border: `1px solid #FDE68A`.
- **Pathology Marker Chip**: High-contrast indicator for detected caries, periapical radiolucencies, or calculus. Soft red ground (`#FEF2F2`), `#DC2626` text, small 6px dot status prefix.

### Data Tables (Periodontal & Diagnostic Charts)
- **Header**: `#F8FAFC` background, `#64748B` label text, uppercase `label-sm` with subtle bottom border `1px solid #E2E8F0`.
- **Row Architecture**: 40px compact height for data density. Alternating row fills are prohibited; differentiation is handled via `1px solid #F1F5F9` borders and `#F8FAFC` hover states.
- **Numeric Cells**: Tabular lining figures, aligned right, with warning color transitions if probing depth (PD) exceeds 4mm (`#EF4444`).

### Interactive Dental Viewer Container
- **Frame**: `#0F172A` (Obsidian) container with `#1E293B` inner frame. Integrated HUD controls for window leveling (contrast/brightness), pan, zoom, and distance calibration in millimeters.
- **Tool Overlays**: Semi-transparent dark slate capsules (`rgba(15, 23, 42, 0.85)`) with backdrop blur (`8px`), hosting white line-art instruments for measuring alveolar bone height or root canal length.

### Form Inputs & Selectors
- **Input Fields**: 40px default height, `#FFFFFF` fill, `1px solid #E2E8F0` border, 8px radius. Active focus transitions border to `#1F4E79` with an optical ring: `0 0 0 3px rgba(31, 78, 121, 0.12)`.
- **Checkboxes & Radios**: 18px square (or circle), 4px border radius for checkbox. Checked state uses `#1F4E79` with a solid white checkmark icon.

### Cards & Modules
- Enclosed with `1px solid #E2E8F0` and `8px` or `12px` radius.
- Cards maintain distinct header zones separated by a hairline border, isolating patient diagnostic metadata from interactive case tools.