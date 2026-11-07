"""Render the reviewed baseline supporting brief. Text is deliberately baseline-specific."""
from pathlib import Path
import json
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from PIL import Image as PILImage

ROOT=Path(__file__).resolve().parent
NAVY=colors.HexColor('#163B56')
BLUE=colors.HexColor('#28649A')
GRAY=colors.HexColor('#52616B')
LIGHT=colors.HexColor('#EDF3F7')
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='TitleCustom',fontName='Helvetica-Bold',fontSize=22,leading=25,textColor=NAVY,spaceAfter=10))
styles.add(ParagraphStyle(name='BodyCustom',fontName='Helvetica',fontSize=10.2,leading=14,textColor=colors.HexColor('#263742'),spaceAfter=8))
styles.add(ParagraphStyle(name='SmallCustom',fontName='Helvetica',fontSize=8.4,leading=11,textColor=GRAY,spaceAfter=8))
styles.add(ParagraphStyle(name='SectionCustom',fontName='Helvetica-Bold',fontSize=12,leading=15,textColor=NAVY,spaceBefore=9,spaceAfter=6))
styles.add(ParagraphStyle(name='TableCustom',fontName='Helvetica',fontSize=9.4,leading=12,textColor=colors.HexColor('#263742')))

def para(text,style='BodyCustom'):
    return Paragraph(text,styles[style])

def table(rows,widths):
    data=[[para(str(c),'TableCustom') for c in row] for row in rows]
    t=Table(data,colWidths=widths,hAlign='LEFT',repeatRows=1)
    t.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),LIGHT),
        ('VALIGN',(0,0),(-1,-1),'TOP'),
        ('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8),
        ('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),
        ('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#B9CBD8')),
        ('LINEBELOW',(0,1),(-1,-1),.35,colors.HexColor('#DDE5EB')),
    ]))
    return t

def figure(name,width=507):
    path=ROOT/'figures'/name
    with PILImage.open(path) as im: w,h=im.size
    return Image(str(path),width=width,height=width*h/w)

def page(c,doc):
    w,h=A4
    c.saveState()
    c.setFillColor(NAVY);c.rect(0,h-12,w,12,fill=1,stroke=0)
    c.setFont('Helvetica-Bold',9);c.drawString(44,h-39,'SSU DATA DASH 2026  /  MEMBER 4')
    c.setFont('Helvetica',8);c.setFillColor(GRAY)
    c.drawRightString(w-44,h-39,'BASELINE SUPPORTING ANALYSIS')
    c.setStrokeColor(colors.HexColor('#DDE5EB'));c.line(44,38,w-44,38)
    c.drawString(44,25,'Prepared 7 November 2026 | Supporting brief, not the final team submission')
    c.drawRightString(w-44,25,str(doc.page))
    c.restoreState()

def main():
    s=json.loads((ROOT/'tables/summary.json').read_text())
    if round(s['off_market']['mdape_pct'],2)!=20.84 or round(s['list_aware']['mdape_pct'],2)!=7.26:
        raise ValueError('This reviewed text is baseline-specific. Update the narrative for new models first.')
    doc=SimpleDocTemplate(str(ROOT/'report/error_analysis_brief.pdf'),pagesize=A4,
                          leftMargin=44,rightMargin=44,topMargin=64,bottomMargin=48,
                          title='SSU Data Dash - Baseline Error Analysis and Uncertainty',author='Member 4 analysis')
    story=[para('Where the baseline fails','TitleCustom'),
           para('Walk-forward predictions, December 2025 to May 2026. Both tracks contain the same <b>10,069 validation homes</b>; 20,138 prediction rows do not represent independent homes.'),
           table([['<b>Track</b>','<b>MdAPE</b>','<b>MAPE</b>','<b>Median signed</b>','<b>90th pct. APE</b>'],
                  ['Off-market','20.84%','27.43%','+8.07%','57.99%'],
                  ['List-aware','7.26%','8.09%','+5.68%','15.89%']],[108,85,85,115,114]),
           Spacer(1,9),figure('01_error_overview.png'),
           para('Figure 1. Positive signed error means overprediction. Sale-price deciles are retrospective groups defined using the outcome. The cumulative-error panel is shown up to 100% APE.','SmallCustom'),
           para('<b>Detached homes deserve priority.</b> Off-market MdAPE is 27.99% for detached homes (n=3,123), versus 16.39% for condo townhouses (n=616). Co-op results are unstable with only 48 homes.'),
           para('<b>Bias differs across prices.</b> The cheapest and dearest sale-price deciles have median signed errors of +31.44% and -36.40%. Conditioning on the outcome can contribute to this pattern; it is not causal evidence. List-aware overpredicts 76.03% of homes.'),
           PageBreak(),
           para('Coverage can hide weak groups','TitleCustom'),
           para('Nominal <b>90% prediction intervals</b> were evaluated on 8,045 eligible homes per track. The other 2,024 homes lacked enough historical residuals: 1,238 December warmup homes plus 786 with insufficient history at listing time.'),
           table([['<b>Measure</b>','<b>Off-market</b>','<b>List-aware</b>'],
                  ['Observed coverage','90.11%','89.36%'],
                  ['Median width / point prediction','114.23%','30.04%'],
                  ['Median width in dollars','$1,067,401','$259,862']],[247,130,130]),
           Spacer(1,10),figure('03_prediction_intervals.png'),
           para('Figure 2. These intervals preserve the baseline point predictions. Width is upper minus lower, divided by the point prediction. Coverage is conditional on calibration eligibility.','SmallCustom'),
           para('A close overall result is not enough','SectionCustom'),
           table([['<b>Property type</b>','<b>Evaluated n</b>','<b>Off-market</b>','<b>List-aware</b>'],
                  ['Detached','2,652','80.28%','86.20%'],
                  ['Condo Apt','3,621','95.44%','92.38%'],
                  ['Condo Townhouse','480','98.13%','91.25%']],[177,110,110,110]),
           Spacer(1,8),
           para('<b>Construction.</b> For each track, use s = |log(sold / predicted)| from sales strictly before both the target listing date and the evaluation month. Require n >= 100; select the k-th ordered score, k = ceil((n+1) x 0.90), then return [prediction x exp(-q), prediction x exp(q)].'),
           para('<b>Interpretation.</b> This historical, conformal-style interval is an empirical backtest, not a strict future-coverage guarantee. Time drift and changing fitted models can violate exchangeability. Wide off-market intervals and low detached coverage motivate testing property-aware calibration on later data.'),
           PageBreak(),
           para('Evidence and next steps','TitleCustom'),
           figure('02_community_errors.png'),
           para('Figure 3. Only communities with at least 50 validation homes are ranked. These exploratory differences are unadjusted for property mix, month and price; names and locations are synthetic.','SmallCustom'),
           para('A reconstructed failure','SectionCustom'),
           para('<b>SYN-0092314:</b> sold for $844,000; predicted $4,317,900; APE 411.60%. Its area was 1,166 sq ft. Rebuilding the official ten-nearest-sales prediction gave the same result. The comparables had a median area of 1,883.5 sq ft; only 3 of 10 shared the community, and sales ranged from March 2021 to September 2025. Nearby, same-type homes can still be poor comparables. This does not isolate the causal effect of any one mismatch.'),
           para('Distinguish score uncertainty from price uncertainty','SectionCustom'),
           table([['<b>95% MdAPE bootstrap range</b>','<b>Off-market</b>','<b>List-aware</b>'],
                  ['Individual-home resampling','20.41-21.36%','7.12-7.42%'],
                  ['Whole-month resampling','19.84-21.93%','7.16-7.40%']],[247,130,130]),
           para('1,000 resamples; seed 42. Six month clusters provide only a rough sensitivity check. These ranges describe the aggregate metric, not uncertainty for an individual house.','SmallCustom'),
           para('Actions for the final model','SectionCustom'),
           para('<b>1.</b> Test size, community and recency adjustments for off-market comparables. <b>2.</b> Test list-price bias correction inside training folds. <b>3.</b> Evaluate property-aware intervals with chronological calibration. None of these improvements has been demonstrated here; rerun the diagnostics on the final model before making performance claims.'),
           para('<b>Provenance and limits.</b> Supplied baseline predictions matched the reported scores. All list-aware predictions equal list price; 24 sampled off-market rows plus 3 extreme cases were reconstructed successfully. Full training provenance was not audited. No hidden-test outcomes or launch-video content were used.','SmallCustom'),
           para('<b>Sources:</b> supplied baseline_results.zip and case handout; <link href="https://huggingface.co/datasets/Baliyang/SSU_data_dash" color="#28649A">official synthetic listings</link>; <link href="https://github.com/erwinzhang7/SSU-Data-Dash" color="#28649A">starter repository</link>; <link href="https://arxiv.org/abs/2107.07511" color="#28649A">Angelopoulos &amp; Bates, conformal prediction introduction</link>. Full tables, input hashes and executable analysis accompany this brief.','SmallCustom')]
    doc.build(story,onFirstPage=page,onLaterPages=page)
    print(ROOT/'report/error_analysis_brief.pdf')

if __name__=='__main__':main()
