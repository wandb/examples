marimo has a rich set of UI components. 

* `mo.ui.altair_chart(altair_chart)` - create a reactive Altair chart
* `mo.ui.button(value=None, kind='primary')` - create a clickable button
* `mo.ui.run_button(label=None, tooltip=None, kind='primary')` - create a button that runs code
* `mo.ui.checkbox(label='', value=False)` - create a checkbox
* `mo.ui.chat(placeholder='', value=None)` - create a chat interface
* `mo.ui.date(value=None, label=None, full_width=False)` - create a date picker
* `mo.ui.dropdown(options, value=None, label=None, full_width=False)` - create a dropdown menu
* `mo.ui.file(label='', multiple=False, full_width=False)` - create a file upload element
* `mo.ui.number(value=None, label=None, full_width=False)` - create a number input
* `mo.ui.radio(options, value=None, label=None, full_width=False)` - create radio buttons
* `mo.ui.refresh(options: List[str], default_interval: str)` - create a refresh control
* `mo.ui.slider(start, stop, value=None, label=None, full_width=False, step=None)` - create a slider
* `mo.ui.range_slider(start, stop, value=None, label=None, full_width=False, step=None)` - create a range slider
* `mo.ui.table(data, columns=None, on_select=None, sortable=True, filterable=True)` - create an interactive table
* `mo.ui.text(value='', label=None, full_width=False)` - create a text input
* `mo.ui.text_area(value='', label=None, full_width=False)` - create a multi-line text input
* `mo.ui.data_explorer(df)` - create an interactive dataframe explorer
* `mo.ui.dataframe(df)` - display a dataframe with search, filter, and sort capabilities
* `mo.ui.plotly(plotly_figure)` - create a reactive Plotly chart (supports scatter, treemap, and sunburst)
* `mo.ui.tabs(elements: dict[str, mo.ui.Element])` - create a tabbed interface from a dictionary
* `mo.ui.array(elements: list[mo.ui.Element])` - create an array of UI elements
* `mo.ui.form(element: mo.ui.Element, label='', bordered=True)` - wrap an element in a form

As always, you can learn more about the available inputs to all these components via `uv --with marimo run python -c "import marimo as mo; help(mo.ui.form)"` 

## Forms

You can compose multiple UI elements into a single form using `.batch().form()`. The `.batch()` method binds named UI elements into a markdown template, and `.form()` adds a submit button so values are only sent on submit.

```python
form = (
    mo.md(
        """
        **Choose an option**

        {choice}

        **Enter some text**

        {text}

        **Enable feature**

        {flag}
        """
    )
    .batch(
        choice=mo.ui.dropdown(options=["A", "B", "C"]),
        text=mo.ui.text(),
        flag=mo.ui.checkbox(),
    )
    .form(
        submit_button_label="Submit",
        show_clear_button=True,   # optional
        clear_on_submit=False,    # keep values after submit
    )
)

form
```

You can also add validation to a form using the `validate` parameter. Return an error string to block submission, or `None` to allow it.

```python
group_by_form = mo.ui.dropdown(
    options=df_columns,
    label="Select column to filter for duplicate analyzis",
    allow_select_none=True,
    value=None,  # start with nothing selected
    searchable=True,
).form(
    submit_button_label="Apply",
    validate=lambda v: (
        "Please select a column and press Apply."
        if v is None else None
    ),
)
```

However, the user may also want to use other components. Popular alternatives include the `ScatterWidget` from the `drawdata` library, `moutils`, and `wigglystuff`. 

For custom classes and static HTML representations you can also use the `_display_` method. 

```python
class Dice:
    def _display_(self):
        import random

        return f"You rolled {random.randint(0, 7)}"
```
