import os
from pathlib import Path
import pymupdf

DOCS_DIR = Path("data/documents")
DOCS_DIR.mkdir(parents=True, exist_ok=True)


def create_pdf(filename: str, pages_content: list[tuple[str, str]]):
    """
    Create a multi-page PDF document using PyMuPDF.
    pages_content is a list of tuples: (heading, body_text)
    """
    filepath = DOCS_DIR / filename
    doc = pymupdf.open()

    for heading, text in pages_content:
        page = doc.new_page(width=595, height=842)  # A4 size
        # Insert Title/Heading
        page.insert_text(
            (50, 60),
            heading,
            fontsize=16,
            fontname="helv",
            color=(0.1, 0.2, 0.6)
        )
        # Horizontal rule
        page.draw_line(pymupdf.Point(50, 75), pymupdf.Point(545, 75), color=(0.7, 0.7, 0.7), width=1)
        
        # Insert Content Body
        rect = pymupdf.Rect(50, 90, 545, 800)
        page.insert_textbox(rect, text, fontsize=10, fontname="helv", color=(0.1, 0.1, 0.1))

    doc.save(str(filepath))
    doc.close()
    print(f"Created: {filepath} ({len(pages_content)} pages)")


def build_all():
    # 1. Academic Regulations and FFCS
    regulations_pages = [
        (
            "VIT-AP University — Academic Regulations & FFCS Guidelines (Page 1)",
            """VIT-AP UNIVERSITY — ACADEMIC REGULATIONS & EVALUATION SYSTEM

1. Fully Flexible Credit System (FFCS)
VIT-AP University implements the Fully Flexible Credit System (FFCS), empowering students to design their own academic trajectories:
- Freedom to choose courses, class timings, and preferred faculty members each semester.
- Opportunity to study at one's own pace (fast-track graduation or spread out credits).
- Flexibility to opt for Minors in complementary disciplines (e.g., Computer Science majors taking a Minor in Business Management or AI) and Honours degrees.
- Minimum credit requirements for B.Tech degree completion: typically 160 to 165 credits spread across 8 semesters.

2. Course Categories
Curricula are divided into:
- University Core (UC): Compulsory foundational subjects including Calculus, Physics, Chemistry, English, and Soft Skills.
- Programme Core (PC): Essential core domain courses for the discipline (e.g., Data Structures, Operating Systems, Computer Networks).
- Programme Electives (PE): Advanced specialized discipline subjects.
- University Electives (UE): Interdisciplinary open electives across engineering, sciences, humanities, and management.
- Co-curricular & Project credits: Capstone project, industrial internships, and technical seminars."""
        ),
        (
            "VIT-AP University — Attendance Policy & Examination Rules (Page 2)",
            """2. Attendance Regulations & Minimum Requirements
VIT-AP University enforces a strict attendance policy to maintain academic rigor:
- Minimum Mandatory Attendance: Students must maintain at least 75% attendance in each registered course (Theory, Lab, and Embedded courses).
- Calculation: Attendance percentage = (Number of classes attended / Total number of classes conducted) * 100.
- Debarment Policy ('N' Grade): Students whose attendance falls below 75% in a course are debarred from appearing in the Term End Examination (TEE) for that course.
- Debarred students receive an 'N' grade and must re-register and re-attend the entire course in a subsequent semester.
- Medical & On-Duty (OD) Leave: Attendance relaxation of up to 10% (lowering threshold to 65%) is considered strictly for authorized medical hospitalization or approved university representation in cultural/sports events, subject to prior approval by the Proctor and Dean of the School.

3. Examination and Evaluation Framework
Evaluation consists of two major components:
- Continuous Assessment Tests (CAT):
  * CAT-1: Conducted around the 5th to 6th week covering initial syllabus modules (typically 15% weightage).
  * CAT-2: Conducted around the 10th to 11th week covering subsequent modules (typically 15% weightage).
  * Digital Assignments & Quizzes: Continuous in-class assessments, quizzes, and project milestones (typically 30% weightage).
- Term End Examination (TEE):
  * Comprehensive final examination conducted at the end of semester covering the complete course syllabus (typically 40% weightage).
  * Eligibility: Mandatory minimum 75% attendance."""
        ),
        (
            "VIT-AP University — Grading System & Academic Standing (Page 3)",
            """4. Grading System and Grade Points
Performance in examinations is translated into letter grades using a 10-point scale:
- S Grade: Outstanding (10 Grade Points)
- A Grade: Excellent (9 Grade Points)
- B Grade: Very Good (8 Grade Points)
- C Grade: Good (7 Grade Points)
- D Grade: Satisfactory (6 Grade Points)
- E Grade: Pass (5 Grade Points)
- F Grade: Fail (0 Grade Points, student must reappear in supplementary/regular exam)
- N Grade: Debarred due to attendance shortage (<75%) (0 Grade Points, student must repeat course)

5. Grade Point Average (GPA) & Cumulative Grade Point Average (CGPA)
- GPA is calculated at the end of each semester:
  GPA = Sum(Course Credits * Grade Points) / Sum(Course Credits)
- CGPA reflects overall cumulative performance across all completed semesters:
  CGPA = Cumulative Sum(Course Credits * Grade Points) / Cumulative Total Credits Registered

6. Academic Integrity & Code of Conduct
- Plagiarism in assignments, laboratory reports, or capstone projects is strictly prohibited.
- Malpractice during CAT or TEE examinations is reviewed by the Examination Disciplinary Committee (EDC) and results in strict penalties including course debarment or suspension."""
        )
    ]
    create_pdf("VITAP_Academic_Regulations_and_FFCS.pdf", regulations_pages)

    # 2. Programs and Admissions Guide
    programs_pages = [
        (
            "VIT-AP University — Academic Programs Guide (Page 1)",
            """VIT-AP UNIVERSITY — ACADEMIC PROGRAMMES & SPECIALIZATIONS

1. Undergraduate Engineering Programmes (B.Tech - 4 Years)
Offered through School of Computer Science and Engineering (SCOPE) and School of Electronics Engineering (SENSE):
- B.Tech Computer Science and Engineering (CSE)
- B.Tech CSE with specialization in Artificial Intelligence and Machine Learning (AI & ML)
- B.Tech CSE with specialization in Data Analytics
- B.Tech CSE with specialization in Cyber Security
- B.Tech CSE with specialization in Software Engineering
- B.Tech CSE with specialization in Block Chain Technology
- B.Tech Electronics and Communication Engineering (ECE)
- B.Tech ECE with specialization in VLSI Design
- B.Tech Mechanical Engineering
- B.Tech Mechanical Engineering with specialization in Electric Vehicles

2. Integrated Dual-Degree Programmes (5 Years)
- Integrated M.Tech Computer Science and Engineering (in collaboration with Virtusa)
- Integrated M.Tech Software Engineering
- Integrated B.Sc. - M.Sc. Data Science
- Integrated B.A. - M.A. Public Policy (5 Years)"""
        ),
        (
            "VIT-AP University — Postgraduate & Doctoral Programmes (Page 2)",
            """3. Postgraduate Programmes
Offered across engineering, sciences, and business:
- M.Tech VLSI Design (School of Electronics Engineering)
- M.Sc. Data Science (School of Advanced Sciences)
- M.Sc. Chemistry
- M.Sc. Physics
- Master of Business Administration (MBA) offered by VIT-AP School of Business (VSB)
- Law Programmes: B.A., LL.B. (Hons.) and BBA., LL.B. (Hons.) offered by VIT-AP School of Law (VSL)

4. Doctoral Programmes (Ph.D.)
Offered as full-time and part-time research fellowships:
- Ph.D. in Computer Science and Engineering
- Ph.D. in Electronics and Communication Engineering
- Ph.D. in Mechanical Engineering
- Ph.D. in Advanced Sciences (Mathematics, Physics, Chemistry)
- Ph.D. in Humanities and Social Sciences (English, Economics)
- Ph.D. in Law and Management

5. Admission & Eligibility Criteria
- B.Tech Admission: Admission is primarily conducted through the VIT Engineering Entrance Examination (VITEEE) on an all-India merit basis.
- Minimum Eligibility: Candidates must have secured an aggregate of at least 60% in Physics, Chemistry, and Mathematics/Biology (50% for SC/ST and North-Eastern state candidates) in the 10+2 qualifying examination.
- Integrated Programmes: Admission based on 10+2 merit scores.
- PG Programmes: Admission based on graduation percentage, entrance tests, and personal interviews."""
        )
    ]
    create_pdf("VITAP_Programs_and_Admissions_Guide.pdf", programs_pages)

    # 3. Campus Facilities and Hostels
    facilities_pages = [
        (
            "VIT-AP University — Campus Facilities & Student Life (Page 1)",
            """VIT-AP UNIVERSITY — CAMPUS FACILITIES, HOSTELS & STUDENT AMENITIES

1. Central Library (Dr. APJ Abdul Kalam Library)
- Spread across a modern multi-floor building with state-of-the-art infrastructure.
- Collections: Extensive collection of textbooks, reference books, national and international journals, and digital e-resources.
- Access to IEEE Xplore, ScienceDirect, Springer, ACM Digital Library, and Scopus databases.
- Automated RFID circulation system with self-service kiosks.
- Air-conditioned silent study halls, discussion rooms, and digital media centres.

2. Hostel Accommodation
VIT-AP offers on-campus residential facilities for both boys (Men's Hostels MH-1 to MH-5) and girls (Ladies' Hostels LH-1 to LH-3):
- Room Types: 2-bed, 3-bed, 4-bed, and 6-bed configurations available with both Air-Conditioned (AC) and Non-AC options.
- Furniture: Individual study table, chair, cot, cupboard, and book rack for each student.
- Wi-Fi: High-speed 24x7 Wi-Fi connectivity throughout residential blocks.
- Security: Round-the-clock biometric security, CCTV surveillance, and dedicated warden care.
- Power: 24/7 power backup through heavy-duty campus generators.

3. Dining & Mess Options
Students choose from diverse hygienic dining options:
- Special Mess: Multi-cuisine menu offering North Indian, South Indian, and continental dishes.
- North Indian Mess & South Indian Mess: Dedicated regional vegetarian and non-vegetarian fare.
- Night Canteens & Food Courts: In-hostel tuck shops, canteens, and branded kiosks operating until late night."""
        ),
        (
            "VIT-AP University — Transport, Sports & Health Services (Page 2)",
            """4. Health and Medical Services
- Health Centre located on campus operates 24x7 with qualified doctors and nursing staff.
- Fully equipped with emergency beds, primary treatment facilities, and a dedicated 24x7 ambulance.
- Formal tie-up with multi-speciality tertiary hospitals in Vijayawada and Guntur for immediate referral care.

5. Sports and Recreation
- Outdoor Facilities: Cricket grounds, football turf, basketball courts with floodlights, volleyball courts, and tennis courts.
- Indoor Sports Complex: Badminton courts, table tennis, chess, and carrom boards.
- Gymnasium: Modern air-conditioned gyms for men and women equipped with cardiovascular and strength-training equipment under professional trainers.

6. Campus Transportation
- University buses operate daily routes connecting the campus to Vijayawada, Guntur, Tenali, and Mangalagiri for day scholars and faculty.
- In-campus battery-operated electric shuttles for eco-friendly internal transport.

7. Innovation and Incubation (IIEC)
- VIT-AP Innovation and Incubation Centre (IIEC) supports student entrepreneurship, patent filings, and seed funding.
- Multiple active student technical chapters (ACM, IEEE, CSI) and cultural clubs (Music, Dance, Dramatics, Photography, Debating)."""
        )
    ]
    create_pdf("VITAP_Campus_Facilities_and_Hostels.pdf", facilities_pages)

    # 4. Fees and Financial Aid
    fees_pages = [
        (
            "VIT-AP University — Tuition Fees & Scholarships Guide (Page 1)",
            """VIT-AP UNIVERSITY — TUITION FEES, HOSTEL EXPENSES & SCHOLARSHIPS

1. Academic Fee Structure (B.Tech Programmes)
Tuition fees for B.Tech programmes are categorized under five tiers based on rank obtained in the VITEEE entrance exam:
- Category 1: Approximately Rs. 1,98,000 per annum (Tuition fee: Rs. 1,73,000 + Caution Deposit: Rs. 3,000 refundable).
- Category 2: Approximately Rs. 3,07,000 per annum.
- Category 3: Approximately Rs. 4,05,000 per annum.
- Category 4: Approximately Rs. 4,48,000 per annum.
- Category 5: Approximately Rs. 4,93,000 per annum.
(Note: Exact fee details vary by academic year and programme discipline as notified in the official admission offer letter.)

2. Hostel and Mess Expenses
- Hostel Room Rent: Ranges from Rs. 60,000 to Rs. 1,40,000 per annum depending on room occupancy (2-bed/3-bed/4-bed) and Air-Conditioning option.
- Mess Charges: Typically ranges between Rs. 55,000 and Rs. 85,000 per annum depending on chosen mess type (Special Mess, Non-Veg Mess, or Veg Mess).
- Caution Deposit: Refundable hostel security deposit collected at the time of admission.

3. Scholarships and Financial Support
- GV School Development Programme (GVSDP):
  * 100% Tuition Fee Waiver for state and central board toppers for all 4 years.
  * VITEEE Rank 1 to 50: 75% tuition fee waiver.
  * VITEEE Rank 51 to 100: 50% tuition fee waiver.
  * VITEEE Rank 101 to 1000: 25% tuition fee waiver.
- STARS (Support The Advancement of Rural Students) Scheme:
  * 100% free education including tuition, hostel, and mess for underprivileged rural students who achieve district top ranks in higher secondary examinations.
- Merit Awards: Semester-wise tuition fee concessions and cash prizes awarded to top 1% academic rankers in each department."""
        )
    ]
    create_pdf("VITAP_Fee_Structure_and_Scholarships.pdf", fees_pages)

    print("All university knowledge base documents generated successfully.")


if __name__ == "__main__":
    build_all()
