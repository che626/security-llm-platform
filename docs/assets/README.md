# Screenshot Assets

These screenshots are used by the GitHub README.

Recommended capture order:

1. `login.png` - product entry and demo account screen.
2. `login-mobile.png` - mobile layout check for the product entry.
3. `workbench.png` - analyst workbench / system home.
4. `dashboard.png` - operational dashboard with demo data loaded.
5. `log-analysis.png` - log analysis evidence and report.
6. `soar.png` - SOAR YAML generation and simulation.
7. `api-docs.png` - FastAPI `/docs` page.

Capture commands can use Playwright after the app is running:

```powershell
npx playwright screenshot --wait-for-selector "text=Security LLM Platform" --wait-for-timeout 1500 --full-page --viewport-size "1440,980" http://localhost:8501 docs/assets/login.png
npx playwright screenshot --wait-for-selector "text=Security LLM Platform" --wait-for-timeout 1500 --viewport-size "390,844" http://localhost:8501 docs/assets/login-mobile.png
```
