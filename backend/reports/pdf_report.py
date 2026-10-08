import io
from typing import Dict, Any
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

class PDFReportGenerator:
    """
    Autonomous PDF Report Generator using ReportLab.
    Produces clean, executive-ready analytical reports.
    """

    def generate(self, report_payload: Dict[str, Any]) -> bytes:
        """Compile report dictionary into formatted PDF bytes."""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=40,
            leftMargin=40,
            topMargin=40,
            bottomMargin=40
        )

        styles = getSampleStyleSheet()
        
        # Custom palette & typography
        primary_color = colors.HexColor("#1E3A8A")   # Deep navy blue
        secondary_color = colors.HexColor("#0D9488") # Teal
        text_dark = colors.HexColor("#1F2937")       # Slate
        bg_light = colors.HexColor("#F3F4F6")        # Light gray
        accent_blue = colors.HexColor("#3B82F6")

        title_style = ParagraphStyle(
            'ReportTitle',
            parent=styles['Heading1'],
            fontName='Helvetica-Bold',
            fontSize=22,
            leading=26,
            textColor=primary_color,
            spaceAfter=6
        )

        subtitle_style = ParagraphStyle(
            'ReportSubtitle',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=11,
            leading=14,
            textColor=colors.HexColor("#4B5563"),
            spaceAfter=15
        )

        h2_style = ParagraphStyle(
            'Heading2_Custom',
            parent=styles['Heading2'],
            fontName='Helvetica-Bold',
            fontSize=13,
            leading=17,
            textColor=primary_color,
            spaceBefore=14,
            spaceAfter=6,
            keepWithNext=True
        )

        body_style = ParagraphStyle(
            'Body_Custom',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=9.5,
            leading=13,
            textColor=text_dark,
            spaceAfter=6
        )

        callout_style = ParagraphStyle(
            'Callout_Custom',
            parent=styles['Normal'],
            fontName='Helvetica-Oblique',
            fontSize=9.5,
            leading=14,
            textColor=colors.HexColor("#1E293B"),
            spaceAfter=4
        )

        bullet_style = ParagraphStyle(
            'Bullet_Custom',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=9,
            leading=13,
            textColor=text_dark,
            leftIndent=15,
            spaceAfter=4
        )

        story = []

        # --- Header Banner ---
        meta = report_payload.get("metadata", {})
        story.append(Paragraph("AUTOANALYST AI", title_style))
        story.append(Paragraph("Autonomous Data Analysis & Business Intelligence Report", subtitle_style))
        story.append(Paragraph(f"<b>Dataset:</b> {meta.get('dataset_name')} | <b>Generated:</b> {meta.get('generated_at')}", body_style))
        story.append(Spacer(1, 10))

        # --- Question Callout Box ---
        question = report_payload.get("4_user_question", "")
        q_data = [
            [Paragraph("<b>INVESTIGATION OBJECTIVE / QUESTION:</b>", ParagraphStyle('QHead', parent=body_style, fontName='Helvetica-Bold', textColor=primary_color))],
            [Paragraph(f"<i>\"{question}\"</i>", callout_style)]
        ]
        q_table = Table(q_data, colWidths=[530])
        q_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), bg_light),
            ('BOX', (0, 0), (-1, -1), 1, accent_blue),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('LEFTPADDING', (0, 0), (-1, -1), 12),
            ('RIGHTPADDING', (0, 0), (-1, -1), 12),
        ]))
        story.append(q_table)
        story.append(Spacer(1, 12))

        # --- 1. Executive Summary ---
        story.append(Paragraph("1. Executive Summary", h2_style))
        story.append(Paragraph(report_payload.get("1_executive_summary", ""), body_style))
        story.append(Spacer(1, 8))

        # --- 2. Key Computed KPIs ---
        story.append(Paragraph("2. Key Computed KPIs (Empirical Results)", h2_style))
        kpis = report_payload.get("7_key_kpis", {})
        kpi_rows = [["Metric Indicator", "Calculated Value", "Benchmark / Context"]]

        pm = kpis.get("primary_metric")
        if pm:
            kpi_rows.append([f"Total {pm['name']}", f"{pm['total']:,}", f"Mean: {pm['mean']:,} | Max: {pm['max']:,}"])

        top = kpis.get("top_performer")
        if top:
            kpi_rows.append([f"Top {top['dimension']}", str(top['category']), f"Total: {top['total']:,} (Mean: {top['mean']:,})"])

        low = kpis.get("lowest_performer")
        if low:
            kpi_rows.append([f"Lowest {low['dimension']}", str(low['category']), f"Total: {low['total']:,} (Mean: {low['mean']:,})"])

        pg = kpis.get("period_growth")
        if pg:
            kpi_rows.append(["Overall Period Growth", f"{pg['overall_growth_pct']}%", f"Recent Period: {pg['recent_period_growth_pct']}%"])

        anom = kpis.get("anomalies")
        if anom:
            kpi_rows.append(["Anomalous Records", f"{anom['count']}", f"{anom['rate_pct']}% of transactions"])

        kpi_table = Table(kpi_rows, colWidths=[180, 150, 200])
        kpi_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), primary_color),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#D1D5DB")),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, bg_light]),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(kpi_table)
        story.append(Spacer(1, 10))

        # --- 3. Important Observed Findings ---
        story.append(Paragraph("3. Important Observed Findings", h2_style))
        for fact in report_payload.get("8_important_findings", []):
            story.append(Paragraph(f"• <b>Observed:</b> {fact}", bullet_style))
        story.append(Spacer(1, 8))

        # --- 4. Statistical Analysis & Hypothesis Testing ---
        story.append(Paragraph("4. Statistical Analysis & Rigor", h2_style))
        stats_notes = report_payload.get("9_statistical_analysis", {}).get("statistical_notes", [])
        for note in stats_notes:
            story.append(Paragraph(f"• <b>Statistical Result:</b> {note}", bullet_style))
        story.append(Spacer(1, 8))

        # --- 5. AI Business Interpretations ---
        story.append(Paragraph("5. Analytical Interpretations", h2_style))
        for interp in report_payload.get("13_business_insights", []):
            story.append(Paragraph(f"• <b>Insight:</b> {interp}", bullet_style))
        story.append(Spacer(1, 8))

        # --- 6. Strategic Recommendations ---
        story.append(Paragraph("6. Actionable Business Recommendations", h2_style))
        for rec in report_payload.get("14_recommendations", []):
            story.append(Paragraph(f"• <b>Action:</b> {rec}", bullet_style))
        story.append(Spacer(1, 8))

        # --- 7. Dataset Overview & Data Quality ---
        story.append(Paragraph("7. Dataset Quality & Audit Overview", h2_style))
        dq = report_payload.get("3_data_quality", {})
        ds = report_payload.get("2_dataset_overview", {})
        dq_text = (
            f"Evaluated <b>{ds.get('rows', 0)}</b> rows and <b>{ds.get('columns', 0)}</b> columns. "
            f"Overall Data Quality Score: <b>{dq.get('quality_score', 100)}/100 ({dq.get('rating')})</b>. "
            f"Missing cells: {dq.get('total_missing_cells', 0)} ({dq.get('missing_percentage', 0)}%), "
            f"Duplicate rows: {dq.get('duplicate_rows', 0)}, Outliers detected: {dq.get('outliers_count', 0)}."
        )
        story.append(Paragraph(dq_text, body_style))
        story.append(Spacer(1, 8))

        # --- 8. Conclusion ---
        story.append(Paragraph("8. Conclusion", h2_style))
        story.append(Paragraph(report_payload.get("15_conclusion", ""), body_style))

        doc.build(story)
        pdf_bytes = buffer.getvalue()
        buffer.close()
        return pdf_bytes
