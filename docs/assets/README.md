# Screenshot Assets

These screenshots are used by the GitHub README.

Recommended capture order:

1. `login.png` - product entry and demo account screen.
2. `workbench.png` - analyst workbench / system home.
3. `dashboard.png` - operational dashboard with demo data loaded.
4. `log-analysis.png` - log analysis evidence and report.
5. `soar.png` - SOAR YAML generation and simulation.
6. `api-docs.png` - FastAPI `/docs` page.

Capture commands can use Playwright after the app is running:

```powershell
npx playwright screenshot --full-page http://localhost:8501 docs/assets/login.png
```
