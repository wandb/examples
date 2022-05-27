import streamlit as st
import altair as alt
import pandas as pd
import numpy as np


def sin_df(parameters:dict) -> pd.DataFrame:
    """
    convert sin numpy array into dataframe format for altair
    parameters has keys:
        times : array of time steps
        amplitude : half the distance between the maximum and minimum values of the function
        offset : vertical shift
        frequency : number of cycles between 0 and 1 (the period is 1/frequency)
        phase : horizontal shift
    """
    times = params['times']
    amp = parameters['amplitude']
    offset = parameters['offset']
    freq = 2 * np.pi * parameters['frequency']
    phase = parameters['phase']
    df = pd.DataFrame({
        'f(t)': amplitude * np.sin(frequency * times + phase) + offset
    })
    return df


def hex_from_name(name):
    """
    convert name representation to a hex string for altair
    """
    if name == 'blue':
        hex_val = '#005b96'
    elif name == 'red':
        hex_val = '#990000'
    return hex_val


def params_to_dict(
    time_params: list,
    amplitude: bool,
    frequency: float,
    phase: float,
    offset: float,
    color: str,
    )-> dict:
    return {
        'times': np.arange(*time_params),
        'amplitude': amplitude,
        'frequency': frequency,
        'phase': phase,
        'offset': offset,
        'color': hex_from_name(color)
    }


# Set sliders & adjustments
time_offset = st.slider(label = 'time offset', min_value = 0., max_value = 10., value = 0.)
amplitude = st.slider(label = 'amplitude', min_value = 0., max_value = 5., value = 1.)
frequency = st.slider(label = 'frequency', min_value = 0., max_value = 1., value = 0.5)
phase = st.slider(label = 'horizontal shift', min_value = 0., max_value = np.pi, value = 0.)
offset = st.slider(label = 'vertical shift', min_value = 0., max_value = 5., value=0.)
color = st.selectbox('line color', ['blue', 'red'])

# convert params to a dictionary with some preprocessing
time_min = 0. + time_offset
time_max = 30. + time_offset
time_step = 0.1
params = params_to_dict(
    [time_min, time_max, time_step],
    amplitude,
    frequency,
    phase,
    offset,
    color)
adjusted_times = params['times'] + time_offset

# construct df of input (time) & output (sin(time)) values
output_df = sin_df(parameters=params)
output_df.index = adjusted_times
output_df = output_df.reset_index().rename(columns={'index':'t', 0:'f(t)'})

# plot df & pass plot handle to streamlit
chart_handle = alt.Chart(output_df).mark_line().encode(
    x=alt.X('t', scale=alt.Scale(domain=[np.min(adjusted_times), np.max(adjusted_times)])),
    y='f(t)',
    color=alt.value(params['color']))
st.altair_chart(chart_handle, use_container_width=True)
