import csv
import html
import os
from typing import List


def generate_benchmark_html(
    *,
    csv_path: str = "benchmarks/benchmarking.csv",
    html_path: str = "benchmarks/index.html",
    title: str = "Model × Task Benchmarks",
) -> None:
    os.makedirs(os.path.dirname(html_path), exist_ok=True)

    rows: List[dict] = []
    if os.path.exists(csv_path):
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

    # Pivot to models as rows and tasks as columns
    models = sorted({r.get("model", "") for r in rows if r.get("model")})
    tasks = sorted({r.get("task", "") for r in rows if r.get("task")})

    values = {}
    for r in rows:
        m = r.get("model", "")
        t = r.get("task", "")
        if not m or not t:
            continue
        values[(m, t)] = (r.get("accuracy", "") or "", r.get("loss", "") or "")

    def esc(s: str) -> str:
        return html.escape(s if s is not None else "")

    head_cells = ["<th data-key=\"model\" onclick=\"sortTable(this)\">Model</th>"]
    for t in tasks:
        head_cells.append(f'<th data-key="{esc(t)}" onclick="sortTable(this)">{esc(t)}</th>')
    thead = "  <tr>" + "".join(head_cells) + "</tr>"

    body_rows = []
    for m in models:
        row_cells = [f'<td data-key="model">{esc(m)}</td>']
        for t in tasks:
            acc, loss = values.get((m, t), ("", ""))
            row_cells.append(
                f'<td data-key="{esc(t)}" data-acc="{esc(acc)}" data-loss="{esc(loss)}" class="num">{esc(acc)}</td>'
            )
        body_rows.append("  <tr>" + "".join(row_cells) + "</tr>")
    tbody = "\n".join(body_rows)

    html_doc = f"""<!doctype html>
<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n<title>{html.escape(title)}</title>\n<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Oxygen, Ubuntu, Cantarell, 'Fira Sans', 'Droid Sans', 'Helvetica Neue', Arial, sans-serif; margin: 24px; }}
h1 {{ font-size: 20px; margin: 0 0 12px; text-align: center; }}
table {{ border-collapse: collapse; width: 90%; margin: 0 auto; }}
th, td {{ padding: 8px 10px; border-bottom: 1px solid #ddd; text-align: center; }}
th {{ cursor: pointer; user-select: none; position: sticky; top: 0; background: #fafafa; }}
tr:hover td {{ background: #f7f7f7; }}
.num {{ text-align: center; font-variant-numeric: tabular-nums; }}
.arrow {{ float: right; opacity: 0.6; }}
</style>\n</head>\n<body>\n<h1>{html.escape(title)}</h1>\n<div id=\"controls\" style=\"margin: 8px 0 16px;\">\n  <label style=\"margin-right:8px;\">Metric:</label>\n  <button id=\"metric-accuracy\" data-metric=\"accuracy\">Accuracy</button>\n  <button id=\"metric-loss\" data-metric=\"loss\">Loss</button>\n</div>\n<table id=\"bench\">\n<thead>\n{thead}\n</thead>\n<tbody>\n{tbody}\n</tbody>\n</table>\n<script>
(function() {{
  function parseNum(v) {{ var x = parseFloat(v); return isNaN(x) ? null : x; }}
  function applyMetric(metric) {{
    var table = document.getElementById('bench');
    var cells = table.querySelectorAll('tbody td[data-acc]');
    cells.forEach(function(td) {{
      var raw = td.getAttribute(metric === 'loss' ? 'data-loss' : 'data-acc') || '';
      if (metric === 'loss') {{
        td.textContent = raw;
      }} else {{
        var x = parseFloat(raw);
        td.textContent = isNaN(x) ? '' : (x * 100).toFixed(1) + '%';
      }}
    }});
  }}

  window.sortTable = function(th) {{
    var table = document.getElementById('bench');
    var key = th.getAttribute('data-key');
    var tbody = table.tBodies[0];
    var rows = Array.prototype.slice.call(tbody.rows);
    var idx = Array.prototype.indexOf.call(th.parentNode.children, th);
    var asc = th.getAttribute('data-asc') !== 'true';
    for (var i=0;i<th.parentNode.children.length;i++) th.parentNode.children[i].removeAttribute('data-asc');
    th.setAttribute('data-asc', asc);
    rows.sort(function(a,b) {{
      var A = a.cells[idx].textContent.trim();
      var B = b.cells[idx].textContent.trim();
      var An = parseNum(A), Bn = parseNum(B);
      if (An !== null && Bn !== null) return asc ? (An - Bn) : (Bn - An);
      return asc ? A.localeCompare(B) : B.localeCompare(A);
    }});
    rows.forEach(function(r) {{ tbody.appendChild(r); }});
  }};
  document.getElementById('metric-accuracy').addEventListener('click', function() {{ applyMetric('accuracy'); }});
  document.getElementById('metric-loss').addEventListener('click', function() {{ applyMetric('loss'); }});
  applyMetric('accuracy');
}})();
</script>\n</body>\n</html>"""

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_doc)
