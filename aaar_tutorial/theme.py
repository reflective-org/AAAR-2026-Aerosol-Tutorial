"""The "reflective" Plotly template: the look of Reflective's presentation figures.

Importing this module registers ``plotly.io.templates["reflective"]`` (config.PLOT_TEMPLATE);
``plots.style()`` applies it. It does not change Plotly's global default template, so figures
that do not go through ``style()`` keep Plotly's own look. It does add FONT_SCRIPT to Plotly's
HTML renderers (the ones ``fig.show()`` uses in Jupyter and Colab): it loads the web fonts
(config.FONT_CSS_URLS) and redraws the figure once they are there, so text is measured in
the real font. For other figures it only loads the fonts (and redraws them once).

    White background; Open Sans text and Gambarino titles, all in Charcoal; left and bottom
    axis lines only, ticks outside, faint grid; frameless legend; thick lines in the brand
    colours (config.SERIES_COLORS); sand buttons and slider with Gold accents.
"""
from __future__ import annotations

import json

import plotly.graph_objects as go
import plotly.io as pio

from . import config


def build_template() -> go.layout.Template:
    """The template as a new object (register() stores it under config.PLOT_TEMPLATE)."""
    ink = config.CHARCOAL
    axis = dict(
        showline=True, linecolor=ink, linewidth=config.PLOT_AXIS_LINE_WIDTH, mirror=False,
        ticks="outside", ticklen=5, tickwidth=config.PLOT_AXIS_LINE_WIDTH, tickcolor=ink,
        tickfont=dict(size=config.PLOT_FONT_SIZE - 0.5, color=ink),
        showgrid=True, gridcolor=config.PLOT_GRID_COLOR, gridwidth=1, zeroline=False,
        title=dict(font=dict(size=config.PLOT_FONT_SIZE + 1.5, color=ink), standoff=12),
        automargin=True)
    colorbar = dict(outlinewidth=0, ticks="outside", ticklen=4, tickcolor=ink, thickness=16,
                    tickfont=dict(color=ink), title=dict(font=dict(color=ink)))
    layout = go.Layout(
        font=dict(family=config.PLOT_FONT_FAMILY, size=config.PLOT_FONT_SIZE, color=ink),
        title=dict(font=dict(family=config.PLOT_TITLE_FONT_FAMILY, size=config.PLOT_TITLE_FONT_SIZE,
                             color=ink),
                   subtitle=dict(font=dict(family=config.PLOT_FONT_FAMILY, size=config.PLOT_FONT_SIZE + 1,
                                           color=ink)),
                   # starts over the left axis line, as in the presentation figures
                   x=0, xref="paper", xanchor="left", y=1, yref="container", yanchor="top",
                   pad=dict(t=16)),
        paper_bgcolor="white", plot_bgcolor="white",
        colorway=list(config.SERIES_COLORS),
        xaxis=axis, yaxis=axis,
        legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0, itemwidth=44,
                    font=dict(size=config.PLOT_FONT_SIZE, color=ink)),
        hoverlabel=dict(bgcolor="white", font=dict(family=config.PLOT_FONT_FAMILY,
                                                   size=config.PLOT_FONT_SIZE - 0.5, color=ink)),
        hovermode="closest",
        bargap=0.25,
        # Linear/Log and play buttons, all Sand: the menus set showactive=False (Plotly would
        # draw the pressed one in a fixed pale blue); the pressed label is bold Charcoal instead.
        updatemenudefaults=dict(bgcolor=config.SAND, bordercolor=config.BEIGE, borderwidth=1,
                                font=dict(size=config.PLOT_FONT_SIZE - 1, color=ink)),
        sliderdefaults=dict(bgcolor=config.SAND, bordercolor=config.GOLD, borderwidth=1,
                            activebgcolor=config.GOLD, ticklen=0, minorticklen=0,
                            tickcolor="rgba(0,0,0,0)", font=dict(color=ink),
                            currentvalue=dict(font=dict(color=ink, size=config.PLOT_FONT_SIZE + 1))),
        annotationdefaults=dict(font=dict(color=ink)),
        coloraxis=dict(colorbar=colorbar),
    )
    data = go.layout.template.Data(
        # One trace template per dash: Plotly cycles them, so lines of your own figures differ in
        # dash as well as colour (traces that set a dash keep it).
        scatter=[go.Scatter(line=dict(width=config.PLOT_LINE_WIDTH, dash=dash),
                            marker=dict(size=config.PLOT_MARKER_SIZE))
                 for dash in config.SERIES_DASHES],
        bar=[go.Bar(marker=dict(line=dict(width=0)))],
        histogram=[go.Histogram(marker=dict(line=dict(width=0)))],
        heatmap=[go.Heatmap(colorbar=colorbar)],
    )
    return go.layout.Template(layout=layout, data=data)


_FONTS_LOADED_MARK = "aaar-webfonts-loaded"     # a font family no system has


def font_script() -> str:
    """JavaScript for a figure's ``post_script``: load the web fonts once per page, then redraw.

    Adds the config.FONT_CSS_URLS stylesheets to the page (unless there already), waits for
    config.FONT_FACES (at most config.FONT_LOAD_TIMEOUT_MS: offline, the fallback fonts stay),
    then redraws the figure, so legend, buttons and slider are measured in the real font.
    ``{plot_id}`` is replaced by Plotly with the figure's div id.
    """
    return f"""(function () {{
  var gd = document.getElementById('{{plot_id}}');
  if (!gd || !window.Plotly || !document.fonts) {{ return; }}
  if (!window.__reflectiveFonts) {{
    window.__reflectiveFonts = new Promise(function (done) {{
      setTimeout(done, {config.FONT_LOAD_TIMEOUT_MS});
      var sheets = {json.dumps(list(config.FONT_CSS_URLS))}.map(function (url) {{
        var found = Array.prototype.some.call(document.querySelectorAll('link[rel="stylesheet"]'),
                                              function (l) {{ return l.href === url; }});
        if (found) {{ return Promise.resolve(); }}
        return new Promise(function (ok) {{
          var link = document.createElement('link');
          link.rel = 'stylesheet'; link.href = url; link.onload = ok; link.onerror = ok;
          document.head.appendChild(link);
        }});
      }});
      Promise.all(sheets).then(function () {{
        return Promise.all({json.dumps(list(config.FONT_FACES))}.map(function (face) {{
          return document.fonts.load(face).catch(function () {{}});
        }}));
      }}).then(done, done);
    }});
  }}
  // Plotly caches text sizes by text and style, so a plain redraw would reuse the fallback
  // font's sizes: append a never-matching family name, which changes the style, not the look.
  window.__reflectiveFonts.then(function () {{
    var full = gd._fullLayout, mark = ', {_FONTS_LOADED_MARK}', update = {{}};
    if (!full || !document.body.contains(gd) || full.font.family.indexOf(mark) >= 0) {{ return; }}
    update['font.family'] = full.font.family + mark;
    if (full.title && full.title.font.family !== full.font.family) {{
      update['title.font.family'] = full.title.font.family + mark;
    }}
    (full.annotations || []).forEach(function (a, i) {{
      if (a.font.family !== full.font.family) {{ update['annotations[' + i + '].font.family'] = a.font.family + mark; }}
    }});
    Plotly.relayout(gd, update);
  }});
}})();"""


FONT_SCRIPT = font_script()


def _add_font_script_to_renderers() -> None:
    """Append FONT_SCRIPT to every Plotly renderer that takes a post_script (once)."""
    for name in pio.renderers:
        renderer = pio.renderers[name]
        if not hasattr(renderer, "post_script"):
            continue
        scripts = renderer.post_script
        scripts = [] if not scripts else [scripts] if isinstance(scripts, str) else list(scripts)
        scripts = [s for s in scripts if "__reflectiveFonts" not in s] + [FONT_SCRIPT]
        renderer.post_script = scripts


def register() -> str:
    """Store the template in plotly.io.templates (again, e.g. after editing config), add the
    font script to Plotly's HTML renderers, and return the template's name."""
    global FONT_SCRIPT
    pio.templates[config.PLOT_TEMPLATE] = build_template()
    FONT_SCRIPT = font_script()
    _add_font_script_to_renderers()
    return config.PLOT_TEMPLATE


register()
