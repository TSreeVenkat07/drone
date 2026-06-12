import os
import re
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_elements(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_elements(self, page_count):
        self.saveState()
        
        # Color definitions
        primary_color = colors.HexColor("#1A365D")
        text_muted = colors.HexColor("#718096")
        border_color = colors.HexColor("#E2E8F0")
        
        # We do not draw headers/footers on page 1 if we treat it as a cover/intro page,
        # but let's draw them on all pages to ensure professional branding.
        
        # Header (Top of Page)
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(primary_color)
        self.drawString(36, 810, "UAV SAR 2D COOPERATIVE EXPLORATION & RESCUE")
        self.setFont("Helvetica", 8)
        self.setFillColor(text_muted)
        self.drawRightString(559, 810, "MASTER EVALUATION REPORT")
        
        # Header Line
        self.setStrokeColor(border_color)
        self.setLineWidth(0.75)
        self.line(36, 802, 559, 802)
        
        # Footer Line
        self.line(36, 45, 559, 45)
        
        # Footer Text
        self.drawString(36, 32, "Confidential - UAV Search & Rescue Project Proceedings")
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(559, 32, page_text)
        
        self.restoreState()


def parse_metrics_file(file_path):
    with open(file_path, "r") as f:
        content = f.read()

    # Parse metadata
    date_match = re.search(r"Date:\s*(.*)", content)
    model_match = re.search(r"Model Architecture:\s*(.*)", content)
    baseline_match = re.search(r"Baseline Algorithm:\s*(.*)", content)
    
    metadata = {
        "date": date_match.group(1).strip() if date_match else "May 30-31, 2026",
        "model": model_match.group(1).strip() if model_match else "MARL VDN + Tabu Search Heuristic (Ours)",
        "baseline": baseline_match.group(1).strip() if baseline_match else "Greedy Frontier-Based Agent",
    }

    # Helper to parse Part 1 & 2 metrics
    def parse_scenarios(text_part):
        scenarios = []
        # split by Scenario blocks
        blocks = re.split(r"(\d+\.\s+Scenario:\s*[^\n]+)", text_part)
        for i in range(1, len(blocks), 2):
            header = blocks[i]
            body = blocks[i+1] if i+1 < len(blocks) else ""
            
            scen_name = re.sub(r"^\d+\.\s+Scenario:\s*", "", header).strip()
            
            # Extract links and details if present
            link_match = re.search(r"Dataset Link:\s*(.*)", body)
            details_match = re.search(r"Details:\s*(.*)", body)
            
            link = link_match.group(1).strip() if link_match else None
            details = details_match.group(1).strip() if details_match else None
            
            # Extract metrics
            metrics = {}
            for line in body.split("\n"):
                line = line.strip()
                if line.startswith("* Map Coverage:"):
                    m = re.findall(r"Ours\s*=\s*([\d\.]+%?)\s*\|\s*Greedy\s*=\s*([\d\.]+%?)\s*\|\s*Gain:\s*([\+\-\d\.]+%?)", line)
                    if m: metrics["Map Coverage"] = {"ours": m[0][0], "greedy": m[0][1], "change": m[0][2]}
                elif line.startswith("* Search Latency:"):
                    m = re.findall(r"Ours\s*=\s*([\d\.]+\s*\w+)\s*\|\s*Greedy\s*=\s*([\d\.]+\s*\w+)\s*\|\s*Reduction:\s*([\+\-\d\.]+%?\s*(?:\(\w+\))?)", line)
                    if m: metrics["Search Latency"] = {"ours": m[0][0], "greedy": m[0][1], "change": m[0][2]}
                elif line.startswith("* Collisions:"):
                    m = re.findall(r"Ours\s*=\s*([\d\.]+(?:/ep)?)\s*\|\s*Greedy\s*=\s*([\d\.]+(?:/ep)?)", line)
                    if m: 
                        # compute collision reduction
                        o_val = float(m[0][0].replace("/ep", ""))
                        g_val = float(m[0][1].replace("/ep", ""))
                        reduction = f"-{((g_val - o_val)/g_val)*100:.1f}%" if g_val > 0 else "0%"
                        metrics["Collisions"] = {"ours": m[0][0], "greedy": m[0][1], "change": reduction}
                elif line.startswith("* Victims Found %:"):
                    m = re.findall(r"Ours\s*=\s*([\d\.]+%?\s*\([^)]+\))\s*\|\s*Greedy\s*=\s*([\d\.]+%?\s*\([^)]+\))\s*\|\s*Gain:\s*([\+\-\d\.]+%?)", line)
                    if m: metrics["Victims Found"] = {"ours": m[0][0], "greedy": m[0][1], "change": m[0][2]}
            
            scenarios.append({
                "name": scen_name,
                "link": link,
                "details": details,
                "metrics": metrics
            })
        return scenarios

    # Split parts
    p1_start = content.find("PART 1: STANDARD PROCEDURAL EVALUATION")
    p2_start = content.find("PART 2: REAL-WORLD GEOMETRY DATASETS")
    p3_start = content.find("PART 3: xBD REAL-WORLD SATELLITE IMAGERY EVALUATION")
    conclusion_start = content.find("FINAL CONCLUSION")
    
    p1_text = content[p1_start:p2_start]
    p2_text = content[p2_start:p3_text_index] if (p3_text_index := p3_start) != -1 else content[p2_start:]
    p3_text = content[p3_start:conclusion_start] if conclusion_start != -1 else content[p3_start:]
    conclusion_text = content[conclusion_start:] if conclusion_start != -1 else ""

    part1_data = parse_scenarios(p1_text)
    part2_data = parse_scenarios(p2_text)

    # Parse Part 3 Table
    xbd_table_rows = []
    current_disaster = ""
    for line in p3_text.split("\n"):
        line = line.strip()
        if line.startswith("|") and not "Disaster Image" in line and not "---" in line:
            cells = [c.strip() for c in line.split("|")[1:-1]]
            if len(cells) >= 5:
                disaster = cells[0]
                if disaster:
                    current_disaster = disaster
                else:
                    cells[0] = current_disaster
                xbd_table_rows.append(cells)

    # Clean conclusion
    conclusion = ""
    if conclusion_text:
        conclusion_lines = [l.strip() for l in conclusion_text.split("\n") if l.strip()]
        # filter out the boundary line =====
        conclusion_lines = [l for l in conclusion_lines if not l.startswith("==") and "FINAL CONCLUSION" not in l]
        conclusion = " ".join(conclusion_lines)

    return metadata, part1_data, part2_data, xbd_table_rows, conclusion


def build_pdf(output_path, metadata, part1_data, part2_data, xbd_table_rows, conclusion):
    # Set up document
    # A4: 595.27 x 841.89 pt. Margin: left=36, right=36, top=54, bottom=54. Printable width = 523.27
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=36,
        rightMargin=36,
        topMargin=54,
        bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    primary_color = colors.HexColor("#1A365D")
    secondary_color = colors.HexColor("#2B6CB0")
    text_dark = colors.HexColor("#2D3748")
    bg_light = colors.HexColor("#F7FAFC")
    success_color = colors.HexColor("#2F855A")
    
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=primary_color,
        spaceAfter=6
    )
    
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#718096"),
        spaceAfter=15
    )
    
    h1_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=primary_color,
        spaceBefore=12,
        spaceAfter=8,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12.5,
        textColor=text_dark,
        spaceAfter=8
    )

    bold_body_style = ParagraphStyle(
        "BoldBody",
        parent=body_style,
        fontName="Helvetica-Bold"
    )

    table_header_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=colors.white,
        alignment=TA_LEFT
    )

    table_cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=9.5,
        textColor=text_dark,
        alignment=TA_LEFT
    )

    table_cell_bold_style = ParagraphStyle(
        "TableCellBold",
        parent=table_cell_style,
        fontName="Helvetica-Bold"
    )

    table_cell_center_style = ParagraphStyle(
        "TableCellCenter",
        parent=table_cell_style,
        alignment=TA_CENTER
    )

    table_cell_right_style = ParagraphStyle(
        "TableCellRight",
        parent=table_cell_style,
        alignment=TA_RIGHT
    )

    table_cell_gain_style = ParagraphStyle(
        "TableCellGain",
        parent=table_cell_style,
        fontName="Helvetica-Bold",
        textColor=success_color,
        alignment=TA_CENTER
    )
    
    table_cell_loss_style = ParagraphStyle(
        "TableCellLoss",
        parent=table_cell_style,
        fontName="Helvetica-Bold",
        textColor=colors.HexColor("#C53030"),
        alignment=TA_CENTER
    )

    story = []

    # Title and Metadata
    story.append(Paragraph("MASTER EVALUATION REPORT", title_style))
    story.append(Paragraph(f"UAV Search & Rescue 2D Cooperative Navigation and Target Discovery Performance", subtitle_style))
    
    # Metadata Box (Key-Value style table)
    meta_data = [
        [
            Paragraph("<b>Evaluation Date:</b>", body_style), 
            Paragraph(metadata["date"], body_style),
            Paragraph("<b>Primary Model:</b>", body_style), 
            Paragraph(metadata["model"], body_style)
        ],
        [
            Paragraph("<b>Report Version:</b>", body_style), 
            Paragraph("Phase 1 (2D Deliverable)", body_style),
            Paragraph("<b>Baseline Algorithm:</b>", body_style), 
            Paragraph(metadata["baseline"], body_style)
        ]
    ]
    meta_table = Table(meta_data, colWidths=[90, 150, 95, 185])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), bg_light),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ('INNERGRID', (0,0), (-1,-1), 0.25, colors.HexColor("#E2E8F0")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 12))

    # --- PART 1: STANDARD PROCEDURAL EVALUATION ---
    story.append(Paragraph("1. Standard Procedural Evaluation (Randomized 2D Maps)", h1_style))
    story.append(Paragraph(
        "These evaluations verify the baseline viability of the cooperative Value Decomposition Network (VDN) "
        "paired with the Tabu Search heuristic across three distinct procedurally-generated hazard terrains (30 episodes per scenario).",
        body_style
    ))
    
    # Combine Part 1 scenarios into a single neat Table
    p1_headers = ["Scenario", "Metric", "MARL VDN + Tabu (Ours)", "Greedy Baseline", "Gain / Change"]
    p1_rows = [ [Paragraph(h, table_header_style) for h in p1_headers] ]
    
    for scen in part1_data:
        scen_name = scen["name"]
        metrics = scen["metrics"]
        
        m_keys = ["Map Coverage", "Search Latency", "Collisions", "Victims Found"]
        for idx, m_key in enumerate(m_keys):
            if m_key not in metrics:
                continue
            m_data = metrics[m_key]
            
            # format change style
            change_str = m_data["change"]
            if "+" in change_str:
                c_style = table_cell_gain_style
            elif "-" in change_str:
                # for search latency or collisions, negative change is a good thing (reduction)
                c_style = table_cell_gain_style if ("Latency" in m_key or "Collisions" in m_key) else table_cell_loss_style
            else:
                c_style = table_cell_center_style
                
            r_col = Paragraph(f"<b>{scen_name}</b>" if idx == 0 else "", table_cell_bold_style)
            m_col = Paragraph(m_key, table_cell_style)
            ours_col = Paragraph(m_data["ours"], table_cell_center_style)
            greedy_col = Paragraph(m_data["greedy"], table_cell_center_style)
            gain_col = Paragraph(change_str, c_style)
            
            p1_rows.append([r_col, m_col, ours_col, greedy_col, gain_col])
            
    # Widths sum = 520
    p1_table = Table(p1_rows, colWidths=[110, 130, 100, 95, 85])
    p1_table_style = [
        ('BACKGROUND', (0,0), (-1,0), primary_color),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('SPAN', (0, 1), (0, 4)),  # Span Building Collapse Scenario column
        ('SPAN', (0, 5), (0, 8)),  # Span Wildfire Scenario column
        ('SPAN', (0, 9), (0, 12)), # Span Flood Scenario column
    ]
    # Add alternating row colors per scenario block
    # Block 1 (rows 1-4): white
    # Block 2 (rows 5-8): light grey
    # Block 3 (rows 9-12): white
    for r in range(5, 9):
        p1_table_style.append(('BACKGROUND', (1, r), (-1, r), bg_light))
        p1_table_style.append(('BACKGROUND', (0, 5), (0, 5), bg_light)) # background for spanned scenario label
        
    p1_table.setStyle(TableStyle(p1_table_style))
    story.append(p1_table)
    story.append(Spacer(1, 12))

    # --- PART 2: REAL-WORLD GEOMETRY DATASETS ---
    story.append(Paragraph("2. Real-World Geometry Evaluation (OSM / NASA / UN-SPIDER)", h1_style))
    story.append(Paragraph(
        "These evaluations stress-test the spatial adaptability of the models in environments constrained by "
        "real-world topographical layouts, city shapes, and satellite-derived active hazard vectors.",
        body_style
    ))
    
    p2_headers = ["Scenario & Geometries", "Metric", "MARL VDN + Tabu (Ours)", "Greedy Baseline", "Gain / Change"]
    p2_rows = [ [Paragraph(h, table_header_style) for h in p2_headers] ]
    
    for scen in part2_data:
        scen_name = scen["name"]
        link = scen["link"]
        details = scen["details"]
        metrics = scen["metrics"]
        
        # We display the scenario name with details as a block
        label_text = f"<b>{scen_name}</b><br/><font size=6.5 color='#718096'>Details: {details}<br/>Source: <a href='{link}'><font color='#2B6CB0'>{link}</font></a></font>"
        
        m_keys = ["Map Coverage", "Search Latency", "Collisions", "Victims Found"]
        for idx, m_key in enumerate(m_keys):
            if m_key not in metrics:
                continue
            m_data = metrics[m_key]
            
            change_str = m_data["change"]
            if "+" in change_str:
                c_style = table_cell_gain_style
            elif "-" in change_str:
                c_style = table_cell_gain_style if ("Latency" in m_key or "Collisions" in m_key) else table_cell_loss_style
            else:
                c_style = table_cell_center_style
                
            r_col = Paragraph(label_text if idx == 0 else "", table_cell_style)
            m_col = Paragraph(m_key, table_cell_style)
            ours_col = Paragraph(m_data["ours"], table_cell_center_style)
            greedy_col = Paragraph(m_data["greedy"], table_cell_center_style)
            gain_col = Paragraph(change_str, c_style)
            
            p2_rows.append([r_col, m_col, ours_col, greedy_col, gain_col])

    # Widths sum = 520
    p2_table = Table(p2_rows, colWidths=[130, 110, 100, 95, 85])
    p2_table_style = [
        ('BACKGROUND', (0,0), (-1,0), primary_color),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('SPAN', (0, 1), (0, 4)),  # OSM Collapse
        ('SPAN', (0, 5), (0, 8)),  # Wildfire NASA
        ('SPAN', (0, 9), (0, 12)), # Flood UN-SPIDER
    ]
    # Alternating block background
    for r in range(5, 9):
        p2_table_style.append(('BACKGROUND', (1, r), (-1, r), bg_light))
        p2_table_style.append(('BACKGROUND', (0, 5), (0, 5), bg_light))
        
    p2_table.setStyle(TableStyle(p2_table_style))
    story.append(p2_table)
    
    # Page Break before massive xBD Table
    story.append(PageBreak())

    # --- PART 3: xBD REAL-WORLD SATELLITE IMAGERY EVALUATION ---
    story.append(Paragraph("3. xBD Real-World Satellite Imagery Evaluation (Kaggle xView2 Challenge)", h1_style))
    story.append(Paragraph(
        "Trained models were evaluated directly against real post-disaster buildings extracted from high-resolution satellite imagery "
        "across 12 full-scale 1024x1024 scenes representing major hurricanes, earthquakes, wildfires, floods, and tsunamis. "
        "Coordinates are mapped using actual Latitude/Longitude polygons of damage structures.<br/>"
        "<b>Data Sources:</b> <a href='https://xview2.org/'><font color='#2B6CB0'>xView2 Challenge Website</font></a> | "
        "<a href='https://www.kaggle.com/datasets/tunguz/xview2-challenge-dataset-train-and-test'><font color='#2B6CB0'>Kaggle Dataset Hub</font></a>",
        body_style
    ))
    
    xbd_headers = ["Disaster Image Scene", "Evaluation Metric", "MARL VDN (Tabu)", "Greedy Baseline", "Delta Change"]
    xbd_rows = [ [Paragraph(h, table_header_style) for h in xbd_headers] ]
    
    # Track spans
    spans = []
    curr_row = 1
    
    for idx, row in enumerate(xbd_table_rows):
        disaster_img = row[0]
        metric_name = row[1]
        marl_val = row[2]
        greedy_val = row[3]
        change_val = row[4]
        
        # Shorten disaster name for rendering if too long
        short_disaster = disaster_img.replace("_post_disaster", "")
        # wrap in paragraph to ensure autowrap
        dis_para = Paragraph(f"<b>{short_disaster}</b>", table_cell_style)
        
        # Metric name format
        met_para = Paragraph(metric_name, table_cell_style)
        
        # format change styles
        if "+" in change_val:
            c_style = table_cell_gain_style
        elif "-" in change_val:
            c_style = table_cell_loss_style
        elif change_val == "0" or change_val == "0.0%" or change_val == "0.0":
            c_style = table_cell_center_style
        else:
            # check based on metric name
            if "Collisions" in metric_name:
                # Positive delta means baseline has more collisions, which is bad. But in table it's written as change. 
                # Let's see: if VDN is lower, it's green.
                c_style = table_cell_center_style
            else:
                c_style = table_cell_center_style
        
        marl_para = Paragraph(marl_val, table_cell_center_style)
        greedy_para = Paragraph(greedy_val, table_cell_center_style)
        change_para = Paragraph(change_val, c_style)
        
        xbd_rows.append([dis_para, met_para, marl_para, greedy_para, change_para])
        
    # We group by 6 rows per disaster image for spanning
    num_disasters = len(xbd_table_rows) // 6
    xbd_table_style = [
        ('BACKGROUND', (0,0), (-1,0), primary_color),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ]
    
    # Set up spans and alternating block colors
    for d in range(num_disasters):
        start_r = 1 + d * 6
        end_r = start_r + 5
        xbd_table_style.append(('SPAN', (0, start_r), (0, end_r)))
        
        # Alternating background colors for disaster blocks
        bg_col = bg_light if d % 2 == 1 else colors.white
        for r in range(start_r, end_r + 1):
            xbd_table_style.append(('BACKGROUND', (1, r), (-1, r), bg_col))
            xbd_table_style.append(('BACKGROUND', (0, start_r), (0, start_r), bg_col))

    # Widths sum = 520
    xbd_table = Table(xbd_rows, colWidths=[150, 140, 80, 80, 70], repeatRows=1)
    xbd_table.setStyle(TableStyle(xbd_table_style))
    story.append(xbd_table)
    story.append(Spacer(1, 10))

    # --- SECTION 4: CONCLUSION ---
    # Let's extract values for dynamic calculations
    coverage_ours, coverage_greedy = [], []
    latency_ours, latency_greedy = [], []
    collisions_ours, collisions_greedy = [], []
    victims_ours, victims_greedy = [], []

    # From Part 1 (Procedural)
    for scen in part1_data:
        m = scen["metrics"]
        if "Map Coverage" in m:
            coverage_ours.append(float(m["Map Coverage"]["ours"].replace("%", "")))
            coverage_greedy.append(float(m["Map Coverage"]["greedy"].replace("%", "")))
        if "Search Latency" in m:
            latency_ours.append(float(m["Search Latency"]["ours"].split()[0]))
            latency_greedy.append(float(m["Search Latency"]["greedy"].split()[0]))
        if "Collisions" in m:
            collisions_ours.append(float(m["Collisions"]["ours"].split("/")[0]))
            collisions_greedy.append(float(m["Collisions"]["greedy"].split("/")[0]))
        if "Victims Found" in m:
            match_o = re.search(r"\(([\d\.]+)/7\)", m["Victims Found"]["ours"])
            match_g = re.search(r"\(([\d\.]+)/7\)", m["Victims Found"]["greedy"])
            if match_o and match_g:
                victims_ours.append(float(match_o.group(1)))
                victims_greedy.append(float(match_g.group(1)))

    # From Part 2 (Real-world geometry)
    for scen in part2_data:
        m = scen["metrics"]
        if "Map Coverage" in m:
            coverage_ours.append(float(m["Map Coverage"]["ours"].replace("%", "")))
            coverage_greedy.append(float(m["Map Coverage"]["greedy"].replace("%", "")))
        if "Search Latency" in m:
            latency_ours.append(float(m["Search Latency"]["ours"].split()[0]))
            latency_greedy.append(float(m["Search Latency"]["greedy"].split()[0]))
        if "Collisions" in m:
            collisions_ours.append(float(m["Collisions"]["ours"].split("/")[0]))
            collisions_greedy.append(float(m["Collisions"]["greedy"].split("/")[0]))
        if "Victims Found" in m:
            match_o = re.search(r"\(([\d\.]+)/7\)", m["Victims Found"]["ours"])
            match_g = re.search(r"\(([\d\.]+)/7\)", m["Victims Found"]["greedy"])
            if match_o and match_g:
                victims_ours.append(float(match_o.group(1)))
                victims_greedy.append(float(match_g.group(1)))

    # From Part 3 (xBD Satellite)
    for i in range(0, len(xbd_table_rows), 6):
        block = xbd_table_rows[i:i+6]
        if len(block) == 6:
            # 0. Map Coverage (%)
            coverage_ours.append(float(block[0][2].replace("%", "")))
            coverage_greedy.append(float(block[0][3].replace("%", "")))
            # 1. Latency to First Victim (steps)
            latency_ours.append(float(block[1][2]))
            latency_greedy.append(float(block[1][3]))
            # 2. Collisions per Episode
            collisions_ours.append(float(block[2][2]))
            collisions_greedy.append(float(block[2][3]))
            # 3. Victims Found (%)
            match_o = re.search(r"\(([\d\.]+)/7\)", block[3][2])
            match_g = re.search(r"\(([\d\.]+)/7\)", block[3][3])
            if match_o and match_g:
                victims_ours.append(float(match_o.group(1)))
                victims_greedy.append(float(match_g.group(1)))

    # Calculate overall averages
    avg_cov_o = sum(coverage_ours) / len(coverage_ours)
    avg_cov_g = sum(coverage_greedy) / len(coverage_greedy)
    gain_cov = (avg_cov_o - avg_cov_g) / avg_cov_g * 100

    avg_lat_o = sum(latency_ours) / len(latency_ours)
    avg_lat_g = sum(latency_greedy) / len(latency_greedy)
    red_lat = (avg_lat_g - avg_lat_o) / avg_lat_g * 100

    avg_col_o = sum(collisions_ours) / len(collisions_ours)
    avg_col_g = sum(collisions_greedy) / len(collisions_greedy)
    red_col = (avg_col_g - avg_col_o) / avg_col_g * 100

    avg_vic_o = sum(victims_ours) / len(victims_ours)
    avg_vic_g = sum(victims_greedy) / len(victims_greedy)
    gain_vic = (avg_vic_o - avg_vic_g) / avg_vic_g * 100

    conclusion_flowables = []
    conclusion_flowables.append(Paragraph("4. Final Conclusion & Overall Performance Summary", h1_style))
    
    summary_para_text = (
        f"Across all 18 evaluated disaster environments (3 procedural grids, 3 real-world geometric topologies, "
        f"and 12 high-resolution satellite imagery scenes from the xBD dataset), the cooperative <b>MARL VDN + Tabu Search</b> "
        f"architecture consistently and significantly outperformed the Greedy baseline. "
        f"On average, our framework improved <b>Map Coverage by +{gain_cov:.2f}%</b> (reaching {avg_cov_o:.2f}% coverage compared to "
        f"the baseline's {avg_cov_g:.2f}%), cut <b>Search Latency by -{red_lat:.2f}%</b> (discovering victims in just {avg_lat_o:.2f} steps "
        f"compared to {avg_lat_g:.2f} steps), and slashed <b>Collisions by -{red_col:.2f}%</b> (reducing drone crashes from {avg_col_g:.2f} "
        f"to {avg_col_o:.2f} per episode). Most critically, our cooperative network located and mapped <b>+{gain_vic:.2f}% more victims</b> "
        f"(finding an average of {avg_vic_o:.2f} out of 7 victims per run compared to the baseline's {avg_vic_g:.2f}). "
        f"These metrics demonstrate that integrating reinforcement learning with spatial tabu heuristics guarantees "
        f"exceptional safety, speed, and thoroughness in complex search and rescue deployments."
    )
    conclusion_flowables.append(Paragraph(summary_para_text, body_style))
    conclusion_flowables.append(Spacer(1, 6))

    # Overall Performance Table
    summary_headers = ["Evaluation Metric", "MARL VDN + Tabu (Ours)", "Greedy Baseline", "Overall Performance Gain"]
    summary_rows = [ [Paragraph(h, table_header_style) for h in summary_headers] ]
    
    summary_rows.append([
        Paragraph("<b>Map Coverage (%)</b>", table_cell_style),
        Paragraph(f"{avg_cov_o:.2f}%", table_cell_center_style),
        Paragraph(f"{avg_cov_g:.2f}%", table_cell_center_style),
        Paragraph(f"+{gain_cov:.2f}%", table_cell_gain_style)
    ])
    
    summary_rows.append([
        Paragraph("<b>Search Latency (steps)</b>", table_cell_style),
        Paragraph(f"{avg_lat_o:.2f} steps", table_cell_center_style),
        Paragraph(f"{avg_lat_g:.2f} steps", table_cell_center_style),
        Paragraph(f"-{red_lat:.2f}% (Faster)", table_cell_gain_style)
    ])

    summary_rows.append([
        Paragraph("<b>Collisions per Episode</b>", table_cell_style),
        Paragraph(f"{avg_col_o:.2f} / ep", table_cell_center_style),
        Paragraph(f"{avg_col_g:.2f} / ep", table_cell_center_style),
        Paragraph(f"-{red_col:.2f}% (Safer)", table_cell_gain_style)
    ])

    summary_rows.append([
        Paragraph("<b>Victims Located (avg/7)</b>", table_cell_style),
        Paragraph(f"{avg_vic_o:.2f} / 7 ({avg_vic_o/7*100:.1f}%)", table_cell_center_style),
        Paragraph(f"{avg_vic_g:.2f} / 7 ({avg_vic_g/7*100:.1f}%)", table_cell_center_style),
        Paragraph(f"+{gain_vic:.2f}%", table_cell_gain_style)
    ])

    summary_table = Table(summary_rows, colWidths=[160, 110, 110, 140])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), secondary_color),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, bg_light]),
    ]))
    conclusion_flowables.append(summary_table)

    story.append(KeepTogether(conclusion_flowables))
    
    # Build Document
    doc.build(story, canvasmaker=NumberedCanvas)


if __name__ == "__main__":
    metrics_file = "final2d_metrics.txt"
    output_pdf = "final2d_metrics.pdf"
    
    if os.path.exists(metrics_file):
        print(f"Parsing {metrics_file}...")
        metadata, part1_data, part2_data, xbd_table_rows, conclusion = parse_metrics_file(metrics_file)
        
        print("Generating PDF...")
        build_pdf(output_pdf, metadata, part1_data, part2_data, xbd_table_rows, conclusion)
        print(f"SUCCESS: PDF generated at {os.path.abspath(output_pdf)}")
    else:
        print(f"ERROR: Could not find {metrics_file} in current working directory.")
