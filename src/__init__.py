"""forecastaudit: scoring, decomposing and recalibrating archived probabilistic forecasts.

The modules are deliberately importable on their own, because each corresponds to one stage of
the pipeline described in the paper: `score` and `score_matched` build the forecast-observation
pairs, `decompose` and `idr_decompose` split the weighted interval score into miscalibration,
discrimination and uncertainty, `idr_exact` bounds the shortcut used by the second of those with
an exact linear program, and `recalibrate` applies conformal correction out of time.
"""

__version__ = "0.1.1"
