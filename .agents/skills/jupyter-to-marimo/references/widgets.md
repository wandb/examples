# Porting ipywidgets to marimo

Jupyter uses **ipywidgets** with imperative callbacks (`observe`, `link`, `jslink`). marimo uses **reactive cells** — a widget's `.value` automatically triggers downstream cells when it changes, so most callback/linking patterns become unnecessary.

## Widget mapping

| ipywidget | marimo | Notes |
| --- | --- | --- |
| `IntSlider` | `mo.ui.slider(start, stop, step=1)` | |
| `FloatSlider` | `mo.ui.slider(start, stop, step=0.1)` | |
| `FloatLogSlider` | `mo.ui.slider(steps=np.logspace(...))` | Use `steps` for non-linear |
| `IntRangeSlider` | `mo.ui.range_slider(start, stop)` | |
| `FloatRangeSlider` | `mo.ui.range_slider(start, stop, step=0.1)` | |
| `IntText` | `mo.ui.number()` | |
| `FloatText` | `mo.ui.number()` | |
| `BoundedIntText` | `mo.ui.number(start, stop)` | |
| `BoundedFloatText` | `mo.ui.number(start, stop)` | |
| `IntProgress` | `mo.status.progress_bar(...)` | Not a UI element; display only |
| `FloatProgress` | `mo.status.progress_bar(...)` | Not a UI element; display only |
| `Checkbox` | `mo.ui.checkbox()` | |
| `ToggleButton` | `mo.ui.switch()` | |
| `Valid` | `mo.md("✓" if valid else "✗")` | No direct equivalent |
| `Dropdown` | `mo.ui.dropdown(options)` | |
| `RadioButtons` | `mo.ui.radio(options)` | |
| `Select` | `mo.ui.dropdown(options)` | |
| `SelectMultiple` | `mo.ui.multiselect(options)` | |
| `SelectionSlider` | `mo.ui.slider(steps=options)` | Use `steps` param |
| `SelectionRangeSlider` | `mo.ui.range_slider(steps=options)` | Use `steps` param |
| `ToggleButtons` | `mo.ui.radio(options, inline=True)` | |
| `Text` | `mo.ui.text()` | |
| `Textarea` | `mo.ui.text_area()` | |
| `Combobox` | `mo.ui.dropdown(options, searchable=True)` | Closest match |
| `Password` | `mo.ui.text(kind="password")` | |
| `Label` | `mo.md("text")` | |
| `HTML` | `mo.Html("...")` | |
| `HTMLMath` | `mo.md(r"$...$")` | See `references/latex.md` |
| `Image` | `mo.image(src)` | |
| `Video` | `mo.video(src)` | |
| `Audio` | `mo.audio(src)` | |
| `DatePicker` | `mo.ui.date()` | |
| `TimePicker` | — | No equivalent; use anywidget |
| `DatetimePicker` | `mo.ui.datetime()` | |
| `NaiveDatetimePicker` | `mo.ui.datetime()` | |
| `ColorPicker` | — | No equivalent; use anywidget |
| `FileUpload` | `mo.ui.file()` | |
| `Button` | `mo.ui.button()` | Use `on_click` or `value` counter pattern |
| `Output` | Cell output / `mo.output.replace()` | See "Output widget" below |
| `Play` | `mo.ui.refresh()` | Periodic refresh, not step-based |
| `TagsInput` | — | No equivalent; use anywidget |
| `ColorsInput` | — | No equivalent; use anywidget |
| `FloatsInput` | — | No equivalent; use anywidget |
| `IntsInput` | — | No equivalent; use anywidget |
| `HBox` | `mo.hstack([...])` | |
| `VBox` | `mo.vstack([...])` | |
| `Box` | `mo.hstack([...])` or `mo.vstack([...])` | |
| `GridBox` | `mo.hstack([...], widths="equal")` | Or use CSS grid |
| `Accordion` | `mo.accordion({...})` | |
| `Tab` | `mo.ui.tabs({...})` | |
| `Stack` | `mo.ui.tabs({...})` or `mo.carousel([...])` | Shows one child at a time |
| `AppLayout` | `mo.sidebar(...)` + stacks | Compose with layout helpers |
| `TwoByTwoLayout` | Nested `mo.vstack`/`mo.hstack` | |
| `GridspecLayout` | CSS grid via `mo.Html` | |
| `Controller` | — | No equivalent; use anywidget |

## Replacing `interact` / `interactive`

ipywidgets `interact` auto-generates UI from a function signature. In marimo, just create the UI elements and use their values. 

Reactivity is automatic:

```python
# Jupyter
from ipywidgets import interact
@interact(x=(0, 10), y=["a", "b", "c"])
def f(x=5, y="a"):
    print(x, y)

# marimo
# cell 1
x = mo.ui.slider(0, 10, value=5)
y = mo.ui.dropdown(["a", "b", "c"], value="a")
mo.hstack([x, y])

# cell 2 — automatically re-runs when x or y change
print(x.value, y.value)
```

## Output widget

Jupyter's `Output` widget captures display output into a container. In marimo, each cell's last expression is its output. For dynamic output:

```python
# Jupyter
out = widgets.Output()
with out:
    print("captured")

# marimo — just use cell output, or:
mo.output.replace(result)
# or redirect stdout:
with mo.redirect_stdout():
    print("goes to cell output")
```

## Replacing `observe` callbacks

ipywidgets use `.observe()` to react to changes. In marimo, split across cells and rely on reactivity:

```python
# Jupyter
slider = widgets.IntSlider(value=5)
output = widgets.Output()
def on_change(change):
    with output:
        output.clear_output()
        print(f"Value: {change['new']}")
slider.observe(on_change, names=['value'])

# marimo
# cell 1
slider = mo.ui.slider(0, 10, value=5)
slider

# cell 2 — automatically re-runs when slider changes
f"Value: {slider.value}"
```

## Replacing `link` / `jslink`

ipywidgets use `link()` or `jslink()` to synchronize widget values. In marimo, use `mo.state` to share state across multiple widgets, or use cell reactivity for directional binding.

### Bidirectional sync via `mo.state` (lifting state up)

Lift shared state into `mo.state` and wire each widget's `on_change` to the setter. This works with native `mo.ui` elements but **not** with anywidgets (use directional binding or `.observe()` for those).

```python
# Jupyter
widgets.jslink((slider, 'value'), (text, 'value'))
```

```python
# marimo — lift state up into mo.state

# cell 1
get_x, set_x = mo.state(0)

# cell 2
x = mo.ui.slider(
    0, 10, value=get_x(), on_change=set_x, label="$x$:"
)

# cell 3
x_plus_one = mo.ui.number(
    1, 11,
    value=get_x() + 1,
    on_change=lambda v: set_x(v - 1),
    label="$x + 1$:",
)

# cell 4
[x, x_plus_one]
```

### Directional binding via cell reactivity

When one widget should drive another (not bidirectional), just read and assign across cells:

```python
# cell 1
slider = mo.ui.slider(0, 10)
counter = Counter(value=0)  # an anywidget
mo.vstack([slider, counter])

# cell 2 — runs when slider changes, updates counter
counter.count = slider.value
```

## Custom widgets / anywidget integration

For ipywidgets with **no marimo equivalent** (marked "—" above), check if the widget is an anywidget or has an anywidget-compatible version. If so, wrap it with `mo.ui.anywidget()`.

If it is not an anywidget, let the user know they should check whether it's a candidate for the [anywidget spec](https://anywidget.dev) — most ipywidgets can be ported. For building custom anywidgets from scratch, invoke the `anywidget-generator` skill.

### Wrapping an existing anywidget

```python
# cell 1
from some_library import CustomWidget
widget = mo.ui.anywidget(CustomWidget(param=42))
widget

# cell 2
widget.value  # dict of all synced traits, reactively updates
```

### Observing individual traits on an anywidget

When you need granular reactivity on specific traits (not the whole `.value` dict), use `mo.state` with `.observe()`:

```python
# cell 1
class Counter(anywidget.AnyWidget):
    _esm = "..."
    _css = "..."
    count = traitlets.Int(0).tag(sync=True)

counter = Counter(count=0)

# create granular state subscriber
get_count, set_count = mo.state(counter.count)
counter.observe(lambda _: set_count(counter.count), names=["count"])

counter

# cell 2
get_count()  # reactively updates when count trait changes
```

## Migration checklist

1. Replace each ipywidget with its marimo equivalent from the table above
2. Remove all `.observe()` callbacks — split logic across reactive cells instead
3. Remove all `link()` / `jslink()` calls — use `mo.state` for bidirectional sync or cell reactivity for directional binding
4. Replace `interact`/`interactive` with explicit `mo.ui` elements
5. Replace `Output` widget with cell output or `mo.output.replace()`
6. Replace layout containers (`HBox`, `VBox`, etc.) with `mo.hstack`, `mo.vstack`, `mo.accordion`, `mo.ui.tabs`
7. For widgets with no equivalent, wrap with `mo.ui.anywidget()` or flag as anywidget candidate
