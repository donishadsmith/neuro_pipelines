from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from bidsaid.logging import setup_logger

LGR = setup_logger(__name__)

TEMPLATE_DIR = Path(__file__).parent / "templates"


class HTMLReport:
    def __init__(self, subject, session, task, analysis_type, method=None):
        self.context = {
            "subject": subject,
            "session": session,
            "task": task,
            "analysis_type": analysis_type,
            "method": method,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.sections = []

    def add_context(self, **kwargs):
        self.context.update(kwargs)

    def mark_excluded(self, reason):
        self.context["excluded"] = True
        self.context["exclusion_reason"] = reason

    def create_report(self, output_path, template_name):
        template = Environment(
            loader=FileSystemLoader(str(TEMPLATE_DIR)),
        ).get_template(template_name)

        html = template.render(sections=self.sections, **self.context)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html, encoding="utf-8")

        LGR.info(f"Report saved to {output_path}")

    @staticmethod
    def append_section(report_path, template_name, context):
        if not report_path.exists():
            return
        
        section = (
            Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)))
            .get_template(template_name)
            .render(**context)
        )

        html = report_path.read_text(encoding="utf-8").replace(
            "</body>", f"\n{section}\n</body>"
        )
        report_path.write_text(html, encoding="utf-8")

        LGR.info(f"Appended {template_name} to {report_path}")
