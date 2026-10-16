"""The "reflective" Plotly template: the chart style of the tutorial web app.

Importing this module registers ``plotly.io.templates["reflective"]`` (config.PLOT_TEMPLATE);
``plots.style()`` applies it. It does not change Plotly's global default template, so figures
that do not go through ``style()`` keep Plotly's own look. It does add FONT_SCRIPT to Plotly's
HTML renderers (the ones ``fig.show()`` uses in Jupyter and Colab): it loads the web fonts
(config.FONT_CSS_URLS) and redraws the figure once they are there, so text is measured in
the real font. For other figures it only loads the fonts (and redraws them once).

    White background; Open Sans, SemiBold Charcoal titles, secondary text (subtitle, ticks,
    legend) in TEXT_2; left and bottom axis lines and short outside ticks in Slate, no grid;
    frameless legend with short samples; thin lines in the brand colours (config.SERIES_COLORS)
    and the app's dashes (config.SERIES_DASHES), no markers; Sand buttons; a white slider rail
    with a Blue outline and a white thumb.
"""
from __future__ import annotations

import json

import plotly.graph_objects as go
import plotly.io as pio

from . import config


def build_template() -> go.layout.Template:
    """The template as a new object (register() stores it under config.PLOT_TEMPLATE)."""
    ink, ink2, axis_ink = config.CHARCOAL, config.TEXT_2, config.AXIS_COLOR
    axis = dict(
        showline=True, linecolor=axis_ink, linewidth=config.PLOT_AXIS_LINE_WIDTH, mirror=False,
        ticks="outside", ticklen=config.PLOT_TICK_LENGTH, tickwidth=config.PLOT_AXIS_LINE_WIDTH,
        tickcolor=axis_ink, ticklabelstandoff=2,
        tickfont=dict(size=config.PLOT_TICK_FONT_SIZE, color=ink2),
        showgrid=False, zeroline=False,
        title=dict(font=dict(size=config.PLOT_AXIS_TITLE_FONT_SIZE, color=ink2), standoff=8),
        automargin=True)
    colorbar = dict(outlinewidth=0, ticks="", thickness=config.HEATMAP_COLORBAR_PX[1],
                    tickfont=dict(color=ink2, size=config.PLOT_NOTE_FONT_SIZE),
                    title=dict(font=dict(color=ink2, size=config.PLOT_NOTE_FONT_SIZE)))
    layout = go.Layout(
        font=dict(family=config.PLOT_FONT_FAMILY, size=config.PLOT_FONT_SIZE, color=ink),
        title=dict(font=dict(family=config.PLOT_TITLE_FONT_FAMILY, size=config.PLOT_TITLE_FONT_SIZE,
                             weight=config.PLOT_TITLE_FONT_WEIGHT, color=ink),
                   subtitle=dict(font=dict(family=config.PLOT_FONT_FAMILY,
                                           size=config.PLOT_SUBTITLE_FONT_SIZE, color=ink2)),
                   # 16 px in from the left edge and 12 px down from the top, as in the app's cards
                   x=0, xref="container", xanchor="left", y=1, yref="container", yanchor="top",
                   # (Plotly draws a two-line title about 6 px above its pad)
                   pad=dict(l=16, t=config.PLOT_TITLE_TOP + 6)),
        paper_bgcolor="white", plot_bgcolor="white",
        colorway=list(config.SERIES_COLORS),
        xaxis=axis, yaxis=axis,
        legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0, itemwidth=30, tracegroupgap=0,
                    font=dict(size=config.PLOT_LEGEND_FONT_SIZE, color=ink2)),
        hoverlabel=dict(bgcolor="white", bordercolor=config.LINE,
                        font=dict(family=config.PLOT_FONT_FAMILY, size=config.PLOT_TICK_FONT_SIZE,
                                  color=ink)),
        hovermode="closest",
        bargap=0.25,
        # Linear/Log and play buttons, Sand: the menus set showactive=False (Plotly would draw
        # the pressed one in a fixed pale blue); the pressed label is bold Charcoal instead.
        updatemenudefaults=dict(bgcolor=config.SAND, bordercolor=config.SAND, borderwidth=1,
                                font=dict(size=config.PLOT_LEGEND_FONT_SIZE, color=ink)),
        # White rail with a Blue outline and a white round thumb with a Blue ring (Plotly draws
        # rail and thumb with the same colours); Sand while pressed.
        sliderdefaults=dict(bgcolor="white", bordercolor=config.BLUE, borderwidth=2,
                            activebgcolor=config.SAND, ticklen=0, minorticklen=0,
                            tickcolor="rgba(0,0,0,0)",
                            font=dict(color=ink2, size=config.PLOT_NOTE_FONT_SIZE),
                            currentvalue=dict(font=dict(color=ink, size=config.PLOT_READOUT_FONT_SIZE,
                                                        family=config.PLOT_MONO_FONT_FAMILY))),
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

    Figures in the template also get the card frame (1 px config.LINE border, rounded corners).
    Adds the config.FONT_CSS_URLS stylesheets to the page (unless there already), waits for
    config.FONT_FACES (at most config.FONT_LOAD_TIMEOUT_MS: offline, the fallback fonts stay),
    then redraws the figure, so legend, buttons and slider are measured in the real font.
    ``{plot_id}`` is replaced by Plotly with the figure's div id.
    """
    return f"""(function () {{
  var gd = document.getElementById('{{plot_id}}');
  if (!gd || !window.Plotly || !document.fonts) {{ return; }}
  // The card frame of the app's charts, on figures in the "reflective" template only (they
  // start with its font family): a style set on the figure's own element (not a stylesheet).
  var family = (gd._fullLayout && gd._fullLayout.font && gd._fullLayout.font.family) || '';
  if (family.indexOf({json.dumps(config.PLOT_FONT_FAMILY)}) === 0 && !gd.dataset.reflectiveCard) {{
    gd.dataset.reflectiveCard = '1';
    gd.style.boxSizing = 'border-box';
    gd.style.border = '1px solid {config.LINE}';
    gd.style.borderRadius = '{config.PLOT_CARD_RADIUS_PX}px';
    gd.style.overflow = 'hidden';
  }}
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
