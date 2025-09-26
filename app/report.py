"""Report generation in multiple formats (HTML, CSV, PDF)."""

import os
import json
import csv
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
import hashlib
import pandas as pd

from .config import config

logger = logging.getLogger(__name__)

class ReportGenerator:
    """Generate comparison reports in multiple formats."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        self.cfg = cfg or config.get('report', {})
        self.template_dir = Path(__file__).parent.parent / "assets"
        self.template_dir.mkdir(exist_ok=True)

    def generate_html_report(self, results: Dict[str, Any],
                           output_path: str) -> bool:
        """Generate comprehensive HTML report."""
        try:
            logger.info(f"Generating HTML report: {output_path}")

            # Load HTML template
            template_path = self.template_dir / "report_template.html"
            if not template_path.exists():
                self._create_default_template()

            with open(template_path, 'r', encoding='utf-8') as f:
                template = f.read()

            # Prepare report data
            report_data = self._prepare_report_data(results)

            # Replace template variables
            html_content = self._render_template(template, report_data)

            # Write to file
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

            logger.info(f"HTML report generated: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to generate HTML report: {e}")
            return False

    def generate_csv_report(self, results: Dict[str, Any],
                          output_path: str) -> bool:
        """Generate CSV report with tabular data."""
        try:
            logger.info(f"Generating CSV report: {output_path}")

            alignment_result = results.get('alignment_result', {})
            similarity_result = results.get('similarity_result', {})

            # Prepare CSV data
            csv_data = []

            # Section-level data
            section_similarities = similarity_result.get('section_similarities', [])
            for section in section_similarities:
                csv_data.append({
                    'Type': 'Section',
                    'ID_A': section.get('section_a_id', ''),
                    'ID_B': section.get('section_b_id', ''),
                    'Title_A': section.get('section_a_title', ''),
                    'Title_B': section.get('section_b_title', ''),
                    'Similarity_%': round(section.get('similarity_score', 0), 2),
                    'Match_Type': section.get('match_type', ''),
                    'Level': section.get('section_level', ''),
                    'Paragraph_Count': section.get('paragraph_count', 0)
                })

            # Paragraph-level data (sample)
            paragraph_alignments = alignment_result.get('paragraph_alignments', [])
            for i, alignment in enumerate(paragraph_alignments[:100]):  # Limit to first 100
                csv_data.append({
                    'Type': 'Paragraph',
                    'ID_A': alignment.get('chunk_a_id', ''),
                    'ID_B': alignment.get('chunk_b_id', ''),
                    'Title_A': alignment.get('chunk_a_text', '')[:100] + '...' if alignment.get('chunk_a_text') else '',
                    'Title_B': alignment.get('chunk_b_text', '')[:100] + '...' if alignment.get('chunk_b_text') else '',
                    'Similarity_%': round(alignment.get('similarity_score', 0) * 100, 2),
                    'Match_Type': alignment.get('alignment_type', ''),
                    'Level': '',
                    'Paragraph_Count': 1
                })

            # Write CSV
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
                writer.writeheader()
                writer.writerows(csv_data)

            logger.info(f"CSV report generated: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to generate CSV report: {e}")
            return False

    def generate_json_report(self, results: Dict[str, Any],
                           output_path: str) -> bool:
        """Generate complete JSON report with all data."""
        try:
            logger.info(f"Generating JSON report: {output_path}")

            # Include metadata
            report_data = {
                'metadata': self._get_report_metadata(),
                'results': results
            }

            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, ensure_ascii=False, indent=2,
                         default=self._json_serializer)

            logger.info(f"JSON report generated: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to generate JSON report: {e}")
            return False

    def generate_pdf_report(self, results: Dict[str, Any],
                          output_path: str) -> bool:
        """Generate PDF report using HTML to PDF conversion."""
        try:
            logger.info(f"Generating PDF report: {output_path}")

            # First generate HTML
            html_path = output_path.replace('.pdf', '_temp.html')
            if not self.generate_html_report(results, html_path):
                return False

            # Convert HTML to PDF
            success = self._html_to_pdf(html_path, output_path)

            # Cleanup temporary HTML
            try:
                os.remove(html_path)
            except:
                pass

            if success:
                logger.info(f"PDF report generated: {output_path}")
            return success

        except Exception as e:
            logger.error(f"Failed to generate PDF report: {e}")
            return False

    def _prepare_report_data(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for template rendering."""
        alignment_result = results.get('alignment_result', {})
        similarity_result = results.get('similarity_result', {})
        processing_info = results.get('processing_info', {})

        # Extract key metrics
        overall_similarity = similarity_result.get('overall_similarity', 0)
        section_similarities = similarity_result.get('section_similarities', [])
        alignment_stats = similarity_result.get('alignment_statistics', {})

        # Create charts data
        section_chart_data = self._create_section_chart_data(section_similarities)
        similarity_distribution = self._create_similarity_distribution(alignment_stats)

        report_data = {
            'title': 'PDF比較レポート',
            'generation_time': datetime.now().strftime('%Y年%m月%d日 %H:%M:%S'),
            'overall_similarity': round(overall_similarity, 2),
            'similarity_grade': self._get_similarity_grade(overall_similarity),
            'total_sections': len(section_similarities),
            'total_paragraphs': alignment_stats.get('total_alignments', 0),
            'processing_time': processing_info.get('total_time_seconds', 0),
            'section_similarities': section_similarities[:10],  # Top 10
            'section_chart_data': section_chart_data,
            'similarity_distribution': similarity_distribution,
            'alignment_statistics': alignment_stats,
            'summary': similarity_result.get('summary', {}),
            'metadata': self._get_report_metadata()
        }

        return report_data

    def _create_section_chart_data(self, section_similarities: List[Dict]) -> Dict[str, Any]:
        """Create data for section similarity chart."""
        if not section_similarities:
            return {'labels': [], 'data': []}

        # Take top 10 sections for chart
        top_sections = section_similarities[:10]

        return {
            'labels': [s.get('section_a_title', 'Section')[:30] for s in top_sections],
            'data': [round(s.get('similarity_score', 0), 1) for s in top_sections]
        }

    def _create_similarity_distribution(self, alignment_stats: Dict) -> Dict[str, Any]:
        """Create similarity distribution for pie chart."""
        distribution = alignment_stats.get('similarity_distribution', {})

        return {
            'labels': ['高い類似度 (>80%)', '中程度 (40-80%)', '低い類似度 (<40%)'],
            'data': [
                distribution.get('high', 0),
                distribution.get('medium', 0),
                distribution.get('low', 0)
            ]
        }

    def _render_template(self, template: str, data: Dict[str, Any]) -> str:
        """Render HTML template with data."""
        # Simple template variable replacement
        for key, value in data.items():
            placeholder = f"{{{{{key}}}}}"

            if isinstance(value, (dict, list)):
                # For complex objects, convert to JSON for JavaScript
                value_str = json.dumps(value, ensure_ascii=False, default=self._json_serializer)
            elif value is None:
                value_str = ""
            else:
                value_str = str(value)

            template = template.replace(placeholder, value_str)

        return template

    def _get_similarity_grade(self, similarity: float) -> str:
        """Get letter grade for similarity score."""
        if similarity >= 90:
            return 'A (優秀)'
        elif similarity >= 80:
            return 'B (良好)'
        elif similarity >= 70:
            return 'C (普通)'
        elif similarity >= 60:
            return 'D (要改善)'
        else:
            return 'F (大幅な相違)'

    def _get_report_metadata(self) -> Dict[str, Any]:
        """Get report metadata."""
        return {
            'generator': 'PDFHikaku v0.1.0',
            'generation_time': datetime.now().isoformat(),
            'config_hash': self._get_config_hash(),
            'environment': {
                'python_version': os.sys.version.split()[0],
                'platform': os.name
            }
        }

    def _get_config_hash(self) -> str:
        """Get hash of current configuration."""
        config_str = json.dumps(dict(config.config_data), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]

    def _json_serializer(self, obj):
        """JSON serializer for complex objects."""
        if hasattr(obj, 'isoformat'):
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            return str(obj)

    def _html_to_pdf(self, html_path: str, pdf_path: str) -> bool:
        """Convert HTML to PDF using available libraries."""
        try:
            # Try weasyprint first
            try:
                import weasyprint
                weasyprint.HTML(filename=html_path).write_pdf(pdf_path)
                return True
            except ImportError:
                pass

            # Try pdfkit
            try:
                import pdfkit
                pdfkit.from_file(html_path, pdf_path)
                return True
            except ImportError:
                pass

            # Fallback: use reportlab to create simple PDF
            return self._create_simple_pdf(pdf_path)

        except Exception as e:
            logger.error(f"HTML to PDF conversion failed: {e}")
            return False

    def _create_simple_pdf(self, output_path: str) -> bool:
        """Create simple PDF using reportlab as fallback."""
        try:
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import letter, A4

            c = canvas.Canvas(output_path, pagesize=A4)
            width, height = A4

            # Simple report content
            c.setFont("Helvetica-Bold", 16)
            c.drawString(50, height - 50, "PDF比較レポート")

            c.setFont("Helvetica", 12)
            c.drawString(50, height - 100, f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            c.drawString(50, height - 130, "詳細な結果はHTMLレポートをご確認ください。")

            c.save()
            return True

        except Exception as e:
            logger.error(f"Simple PDF creation failed: {e}")
            return False

    def _create_default_template(self):
        """Create default HTML template."""
        template_path = self.template_dir / "report_template.html"

        template_content = """<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{title}}</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 2px solid #eee;
        }
        .header h1 {
            color: #2c3e50;
            margin-bottom: 10px;
        }
        .summary {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .summary-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 6px;
            text-align: center;
            border-left: 4px solid #3498db;
        }
        .summary-card h3 {
            margin: 0 0 10px 0;
            color: #2c3e50;
        }
        .summary-card .value {
            font-size: 24px;
            font-weight: bold;
            color: #3498db;
        }
        .section {
            margin-bottom: 30px;
        }
        .section h2 {
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }
        .section-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }
        .section-table th,
        .section-table td {
            text-align: left;
            padding: 12px;
            border-bottom: 1px solid #ddd;
        }
        .section-table th {
            background-color: #3498db;
            color: white;
        }
        .section-table tr:hover {
            background-color: #f5f5f5;
        }
        .similarity-high { color: #27ae60; font-weight: bold; }
        .similarity-medium { color: #f39c12; font-weight: bold; }
        .similarity-low { color: #e74c3c; font-weight: bold; }
        .metadata {
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #eee;
            font-size: 12px;
            color: #666;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{{title}}</h1>
            <p>生成日時: {{generation_time}}</p>
        </div>

        <div class="summary">
            <div class="summary-card">
                <h3>総合類似度</h3>
                <div class="value">{{overall_similarity}}%</div>
                <div>評価: {{similarity_grade}}</div>
            </div>
            <div class="summary-card">
                <h3>比較セクション数</h3>
                <div class="value">{{total_sections}}</div>
            </div>
            <div class="summary-card">
                <h3>比較段落数</h3>
                <div class="value">{{total_paragraphs}}</div>
            </div>
            <div class="summary-card">
                <h3>処理時間</h3>
                <div class="value">{{processing_time}}秒</div>
            </div>
        </div>

        <div class="section">
            <h2>📊 セクション別類似度</h2>
            <table class="section-table">
                <thead>
                    <tr>
                        <th>セクション名</th>
                        <th>類似度</th>
                        <th>マッチタイプ</th>
                        <th>段落数</th>
                    </tr>
                </thead>
                <tbody id="sections-table">
                    <!-- Sections will be inserted here -->
                </tbody>
            </table>
        </div>

        <div class="metadata">
            <h3>レポート情報</h3>
            <p>生成システム: {{metadata.generator}}</p>
            <p>設定ハッシュ: {{metadata.config_hash}}</p>
        </div>
    </div>

    <script>
        // Insert section data
        const sections = {{section_similarities}};
        const tbody = document.getElementById('sections-table');

        sections.forEach(section => {
            const row = tbody.insertRow();
            const similarity = section.similarity_score || 0;

            let similarityClass = 'similarity-low';
            if (similarity >= 80) similarityClass = 'similarity-high';
            else if (similarity >= 40) similarityClass = 'similarity-medium';

            row.innerHTML = `
                <td>${section.section_a_title || 'Unknown'}</td>
                <td class="${similarityClass}">${similarity.toFixed(1)}%</td>
                <td>${section.match_type || '-'}</td>
                <td>${section.paragraph_count || 0}</td>
            `;
        });
    </script>
</body>
</html>"""

        with open(template_path, 'w', encoding='utf-8') as f:
            f.write(template_content)

        logger.info(f"Created default HTML template: {template_path}")

class MultiFormatReporter:
    """Generate reports in multiple formats simultaneously."""

    def __init__(self):
        self.generator = ReportGenerator()

    def generate_all_reports(self, results: Dict[str, Any],
                           base_path: str,
                           formats: List[str] = None) -> Dict[str, bool]:
        """Generate reports in all requested formats."""
        if formats is None:
            formats = ['html', 'csv', 'json']

        report_results = {}
        base_name = Path(base_path).stem

        for format_type in formats:
            output_path = f"{base_name}.{format_type}"

            if format_type == 'html':
                success = self.generator.generate_html_report(results, output_path)
            elif format_type == 'csv':
                success = self.generator.generate_csv_report(results, output_path)
            elif format_type == 'json':
                success = self.generator.generate_json_report(results, output_path)
            elif format_type == 'pdf':
                success = self.generator.generate_pdf_report(results, output_path)
            else:
                logger.warning(f"Unsupported format: {format_type}")
                success = False

            report_results[format_type] = success

        return report_results