from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


OUTPUT_DIR = Path(r"C:\programming\code\article_python\output\literature")
BIB_PATH = OUTPUT_DIR / "初稿6_最终34篇文献_Zotero导入.bib"
RIS_PATH = OUTPUT_DIR / "初稿6_最终34篇文献_Zotero导入.ris"
SUMMARY_PATH = OUTPUT_DIR / "初稿6_最终34篇文献_Zotero导入说明.md"
ZIP_PATH = OUTPUT_DIR / "初稿6_最终34篇文献_Zotero导入包.zip"


@dataclass(frozen=True)
class Record:
    key: str
    kind: str
    authors: list[str]
    title: str
    year: int
    venue: str
    volume: str = ""
    number: str = ""
    pages: str = ""
    doi: str = ""
    url: str = ""
    publisher: str = ""
    bib_type: str = "article"
    ris_type: str = "JOUR"
    extra: dict[str, str] = field(default_factory=dict)


RECORDS = [
    Record("Aggarwal2020Path", "journal", ["Aggarwal, Shubhani", "Kumar, Neeraj"], "Path planning techniques for unmanned aerial vehicles: A review, solutions, and challenges", 2020, "Computer Communications", volume="149", pages="270--299", doi="10.1016/j.comcom.2019.10.014", url="https://doi.org/10.1016/j.comcom.2019.10.014"),
    Record("Ali2023Feature", "journal", ["Ali, Hub", "Xiong, Gang", "Haider, Muhammad Husnain", "Tamir, Tariku Sinshaw", "Dong, Xisong", "Shen, Zhen"], "Feature selection-based decision model for UAV path planning on rough terrains", 2023, "Expert Systems with Applications", volume="232", pages="120713", doi="10.1016/j.eswa.2023.120713", url="https://doi.org/10.1016/j.eswa.2023.120713"),
    Record("Bertolotto2020Volunteered", "journal", ["Bertolotto, Michela", "McArdle, Gavin", "Schoen-Phelan, Bianca"], "Volunteered and crowdsourced geographic information: The OpenStreetMap project", 2020, "Journal of Spatial Information Science", number="20", doi="10.5311/josis.2020.20.659", url="https://doi.org/10.5311/josis.2020.20.659"),
    Record("Bhuiyan2024Aerial", "journal", ["Bhuiyan, Tanveer Hossain", "Walker, Victor", "Roni, Mohammad", "Ahmed, Ishfaq"], "Aerial drone fleet deployment optimization with endogenous battery replacements for direct delivery of time-sensitive products", 2024, "Expert Systems with Applications", volume="252", pages="124172", doi="10.1016/j.eswa.2024.124172", url="https://doi.org/10.1016/j.eswa.2024.124172"),
    Record("Chen2023BSpline", "conference", ["Chen, Wantong", "Yang, Qianqian", "Diao, Tianru", "Ren, Shuzhen"], "B-spline fusion line of sight algorithm for UAV path planning", 2023, "Lecture Notes in Electrical Engineering", pages="503--512", doi="10.1007/978-981-19-6613-2_50", url="https://doi.org/10.1007/978-981-19-6613-2_50", bib_type="incollection", ris_type="CHAP"),
    Record("Cichocinski2021Study", "journal", ["Cichociński, Piotr"], "A study on the usability of open spatial data for road network-based analysis: Using OpenStreetMap as an example", 2021, "Geoinformatica Polonica", volume="20", pages="89--96", doi="10.4467/21995923GP.21.007.14978", url="https://doi.org/10.4467/21995923GP.21.007.14978"),
    Record("Fan2023UAV", "journal", ["Fan, Jiaming", "Chen, Xia", "Liang, Xiao"], "UAV trajectory planning based on bi-directional APF-RRT* algorithm with goal-biased", 2023, "Expert Systems with Applications", volume="213", pages="119137", doi="10.1016/j.eswa.2022.119137", url="https://doi.org/10.1016/j.eswa.2022.119137"),
    Record("Ghambari2024UAV", "journal", ["Ghambari, Soheila", "Golabi, Mahmoud", "Jourdan, Laetitia", "Lepagnot, Julien", "Idoumghar, Lhassane"], "UAV path planning techniques: A survey", 2024, "RAIRO - Operations Research", volume="58", number="4", pages="2951--2989", doi="10.1051/ro/2024073", url="https://doi.org/10.1051/ro/2024073"),
    Record("Hart1968Formal", "journal", ["Hart, Peter", "Nilsson, Nils", "Raphael, Bertram"], "A formal basis for the heuristic determination of minimum cost paths", 1968, "IEEE Transactions on Systems Science and Cybernetics", volume="4", number="2", pages="100--107", doi="10.1109/TSSC.1968.300136", url="https://doi.org/10.1109/TSSC.1968.300136"),
    Record("Hu2025Layered", "journal", ["Hu, Xiao-Bing", "Yang, Chang-Shu", "Zhou, Jun", "Zhang, Ying-Fei", "Ma, Yi-Ming"], "Research on 3D layered visibility graph route network model and multi-objective path planning for UAVs in complex urban environments", 2025, "Aerospace Science and Technology", volume="159", pages="109947", doi="10.1016/j.ast.2025.109947", url="https://doi.org/10.1016/j.ast.2025.109947"),
    Record("Huang2021Path", "journal", ["Huang, Hailong", "Savkin, Andrey V."], "Path planning for a solar-powered UAV inspecting mountain sites for safety and rescue", 2021, "Energies", volume="14", number="7", pages="1968", doi="10.3390/en14071968", url="https://doi.org/10.3390/en14071968"),
    Record("Karaman2011Sampling", "journal", ["Karaman, Sertac", "Frazzoli, Emilio"], "Sampling-based algorithms for optimal motion planning", 2011, "The International Journal of Robotics Research", volume="30", number="7", pages="846--894", doi="10.1177/0278364911406761", url="https://doi.org/10.1177/0278364911406761"),
    Record("Koenig2002DLite", "conference", ["Koenig, Sven", "Likhachev, Maxim"], "D* Lite", 2002, "Proceedings of the 18th AAAI Conference on Artificial Intelligence (AAAI '02)", pages="476--483", url="https://aaai.org/papers/00476-aaai02-072-d-lite/", publisher="AAAI Press", bib_type="inproceedings", ris_type="CONF"),
    Record("Koenig2004Lifelong", "journal", ["Koenig, Sven", "Likhachev, Maxim", "Furcy, David"], "Lifelong Planning A*", 2004, "Artificial Intelligence", volume="155", number="1-2", pages="93--146", doi="10.1016/j.artint.2003.12.001", url="https://doi.org/10.1016/j.artint.2003.12.001"),
    Record("Liu2025Robust", "conference", ["Liu, Junjie", "Zou, Yao"], "Robust UAV path planning via safe flight corridor and penalty function", 2025, "2025 40th Youth Academic Annual Conference of Chinese Association of Automation (YAC)", pages="2138--2143", doi="10.1109/YAC66630.2025.11150073", url="https://doi.org/10.1109/YAC66630.2025.11150073", bib_type="inproceedings", ris_type="CONF"),
    Record("Margraff2020UAV", "conference", ["Margraff, Julien", "Stephant, Joanny", "Labbani-Igbida, Ouiddad"], "UAV 3D path and motion planning in unknown dynamic environments", 2020, "2020 International Conference on Unmanned Aircraft Systems (ICUAS)", pages="77--84", doi="10.1109/ICUAS48674.2020.9214057", url="https://doi.org/10.1109/ICUAS48674.2020.9214057", bib_type="inproceedings", ris_type="CONF"),
    Record("Mishra2024Integrated", "journal", ["Mishra, Dev", "Tiwari, Manoj Kumar"], "Integrated truck drone delivery services with an optimal charging stations", 2024, "Expert Systems with Applications", volume="254", pages="124254", doi="10.1016/j.eswa.2024.124254", url="https://doi.org/10.1016/j.eswa.2024.124254"),
    Record("Mohammed2021Line", "conference", ["Mohammed, Imran", "Collings, Iain B.", "Hanly, Stephen V."], "Line of sight probability prediction for UAV communication", 2021, "2021 IEEE International Conference on Communications Workshops (ICC Workshops)", pages="1--6", doi="10.1109/ICCWorkshops50388.2021.9473740", url="https://doi.org/10.1109/ICCWorkshops50388.2021.9473740", bib_type="inproceedings", ris_type="CONF"),
    Record("MoshrefJavadi2021Applications", "journal", ["Moshref-Javadi, Mohammad", "Winkenbach, Matthias"], "Applications and research avenues for drone-based models in logistics: A classification and review", 2021, "Expert Systems with Applications", volume="177", pages="114854", doi="10.1016/j.eswa.2021.114854", url="https://doi.org/10.1016/j.eswa.2021.114854"),
    Record("Nikolaiev2025Enhanced", "journal", ["Nikolaiev, Mykola", "Novotarskyi, Mykhailo"], "An enhanced adaptive B-spline smoothing approach for UAV path planning", 2025, "International Journal of Intelligent Systems and Applications", volume="17", number="4", pages="1--13", doi="10.5815/ijisa.2025.04.01", url="https://doi.org/10.5815/ijisa.2025.04.01"),
    Record("Ramos2023Hybrid", "journal", ["Ramos, Tânia Rodrigues Pereira", "Vigo, Daniele"], "A new hybrid distribution paradigm: Integrating drones in medicines delivery", 2023, "Expert Systems with Applications", volume="234", pages="120992", doi="10.1016/j.eswa.2023.120992", url="https://doi.org/10.1016/j.eswa.2023.120992"),
    Record("Salhi2023Multiobjective", "conference", ["Salhi, M.", "Delavernhe, F."], "Multiobjective UAV path planning: Connectivity quality and energy consumption", 2023, "GLOBECOM 2023 - 2023 IEEE Global Communications Conference", pages="746--751", doi="10.1109/GLOBECOM54140.2023.10437552", url="https://doi.org/10.1109/GLOBECOM54140.2023.10437552", bib_type="inproceedings", ris_type="CONF"),
    Record("Shao2025Energy", "journal", ["Shao, Qi", "Mao, Xuefei", "Xu, Wenbin"], "Energy-aware UAV coverage planning in mountainous terrain via contour-aligned path generation", 2025, "IEEE Robotics and Automation Letters", volume="10", number="12", pages="12373--12380", doi="10.1109/LRA.2025.3621932", url="https://doi.org/10.1109/LRA.2025.3621932"),
    Record("Simplicio2025MultiResolution", "conference", ["Simplicio, Paulo V. G.", "Pereira, Guilherme A. S."], "Multi-resolution UAV path replanning for inspection of tailings dams", 2025, "2025 International Conference on Unmanned Aircraft Systems (ICUAS)", pages="256--263", doi="10.1109/ICUAS65942.2025.11007892", url="https://doi.org/10.1109/ICUAS65942.2025.11007892", bib_type="inproceedings", ris_type="CONF"),
    Record("Stentz1994Optimal", "conference", ["Stentz, A."], "Optimal and efficient path planning for partially-known environments", 1994, "Proceedings of the 1994 IEEE International Conference on Robotics and Automation", pages="3310--3317", doi="10.1109/ROBOT.1994.351061", url="https://doi.org/10.1109/ROBOT.1994.351061", bib_type="inproceedings", ris_type="CONF"),
    Record("Wang2024UAV", "journal", ["Wang, Wentao", "Li, Xiaoli", "Tian, Jun"], "UAV formation path planning for mountainous forest terrain utilizing an artificial rabbit optimizer incorporating reinforcement learning and thermal conduction search strategies", 2024, "Advanced Engineering Informatics", volume="62", pages="102947", doi="10.1016/j.aei.2024.102947", url="https://doi.org/10.1016/j.aei.2024.102947"),
    Record("Wu2022Optimal", "journal", ["Wu, Min", "Chen, Wuhua", "Tian, Xiaohong"], "Optimal energy consumption path planning for quadrotor UAV transmission tower inspection based on simulated annealing algorithm", 2022, "Energies", volume="15", number="21", pages="8036", doi="10.3390/en15218036", url="https://doi.org/10.3390/en15218036"),
    Record("Xu2023Cooperative", "journal", ["Xu, Liang", "Cao, Xianbin", "Du, Wenbo", "Li, Yong"], "Cooperative path planning optimization for multiple UAVs with communication constraints", 2023, "Knowledge-Based Systems", volume="260", pages="110164", doi="10.1016/j.knosys.2022.110164", url="https://doi.org/10.1016/j.knosys.2022.110164"),
    Record("Yilmaz2024Evolutionary", "journal", ["Yılmaz, Cemal", "Cengiz, Enes", "Kahraman, Hamdi Tolga"], "A new evolutionary optimization algorithm with hybrid guidance mechanism for truck-multi drone delivery system", 2024, "Expert Systems with Applications", volume="245", pages="123115", doi="10.1016/j.eswa.2023.123115", url="https://doi.org/10.1016/j.eswa.2023.123115"),
    Record("Yu2023Hybrid", "journal", ["Yu, Xiaobing", "Jiang, Nijun", "Wang, Xuming", "Li, Meijuan"], "A hybrid algorithm based on grey wolf optimizer and differential evolution for UAV path planning", 2023, "Expert Systems with Applications", volume="215", pages="119327", doi="10.1016/j.eswa.2022.119327", url="https://doi.org/10.1016/j.eswa.2022.119327"),
    Record("YuLuo2023Reinforcement", "journal", ["Yu, Xiaobing", "Luo, Wenguan"], "Reinforcement learning-based multi-strategy cuckoo search algorithm for 3D UAV path planning", 2023, "Expert Systems with Applications", volume="223", pages="119910", doi="10.1016/j.eswa.2023.119910", url="https://doi.org/10.1016/j.eswa.2023.119910"),
    Record("Zhang2023Novel", "journal", ["Zhang, Chaoqun", "Zhou, Wenjuan", "Qin, Weidong", "Tang, Wen"], "A novel UAV path planning approach: Heuristic crossing search and rescue optimization algorithm", 2023, "Expert Systems with Applications", volume="215", pages="119243", doi="10.1016/j.eswa.2022.119243", url="https://doi.org/10.1016/j.eswa.2022.119243"),
    Record("Zhang2024MultiObjective", "journal", ["Zhang, Wenhui", "Peng, Chaoda", "Yuan, Yuan", "Cui, Jian", "Qi, Lin"], "A novel multi-objective evolutionary algorithm with a two-fold constraint-handling mechanism for multiple UAV path planning", 2024, "Expert Systems with Applications", volume="238", pages="121862", doi="10.1016/j.eswa.2023.121862", url="https://doi.org/10.1016/j.eswa.2023.121862"),
    Record("Zhou2024State", "journal", ["Zhou, Xiaojun", "Tang, Zhouhang", "Wang, Nan", "Yang, Chao", "Huang, Tianxiang"], "A novel state transition algorithm with adaptive fuzzy penalty for multi-constraint UAV path planning", 2024, "Expert Systems with Applications", volume="248", pages="123481", doi="10.1016/j.eswa.2024.123481", url="https://doi.org/10.1016/j.eswa.2024.123481"),
]


def protect_bibtex_title(title: str) -> str:
    protected_terms = {
        "UAV": "{UAV}",
        "OSM": "{OSM}",
        "OpenStreetMap": "{OpenStreetMap}",
        "RRT*": "{RRT*}",
        "APF-RRT*": "{APF-RRT*}",
        "D* Lite": "{D* Lite}",
        "A*": "{A*}",
        "LPA*": "{LPA*}",
        "B-spline": "{B}-spline",
        "3D": "{3D}",
        "AAAI": "{AAAI}",
    }
    result = title
    for raw, protected in protected_terms.items():
        result = result.replace(raw, protected)
    return result


def bibtex_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")


def bibtex_entry(record: Record) -> str:
    fields = [
        ("author", " and ".join(record.authors)),
        ("title", protect_bibtex_title(record.title)),
        ("year", str(record.year)),
    ]
    if record.bib_type in {"article"}:
        fields.append(("journal", record.venue))
    elif record.bib_type == "incollection":
        fields.append(("booktitle", record.venue))
    else:
        fields.append(("booktitle", record.venue))
    if record.volume:
        fields.append(("volume", record.volume))
    if record.number:
        fields.append(("number", record.number))
    if record.pages:
        fields.append(("pages", record.pages))
    if record.publisher:
        fields.append(("publisher", record.publisher))
    if record.doi:
        fields.append(("doi", record.doi))
    if record.url:
        fields.append(("url", record.url))

    lines = [f"@{record.bib_type}{{{record.key},"]
    for index, (field, value) in enumerate(fields):
        comma = "," if index < len(fields) - 1 else ""
        lines.append(f"  {field} = {{{value}}}{comma}")
    lines.append("}")
    return "\n".join(lines)


def ris_entry(record: Record) -> str:
    lines = [f"TY  - {record.ris_type}"]
    for author in record.authors:
        lines.append(f"AU  - {author}")
    lines.extend(
        [
            f"PY  - {record.year}",
            f"TI  - {record.title}",
        ]
    )
    if record.ris_type in {"CONF", "CHAP"}:
        lines.append(f"T2  - {record.venue}")
    else:
        lines.append(f"JO  - {record.venue}")
        lines.append(f"JF  - {record.venue}")
    if record.volume:
        lines.append(f"VL  - {record.volume}")
    if record.number:
        lines.append(f"IS  - {record.number}")
    if record.pages:
        lines.append(f"SP  - {record.pages.replace('--', '-')}")
    if record.publisher:
        lines.append(f"PB  - {record.publisher}")
    if record.doi:
        lines.append(f"DO  - {record.doi}")
    if record.url:
        lines.append(f"UR  - {record.url}")
    lines.append("ER  - ")
    return "\n".join(lines)


def write_outputs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    BIB_PATH.write_text("\n\n".join(bibtex_entry(record) for record in RECORDS) + "\n", encoding="utf-8")
    RIS_PATH.write_text("\n\n".join(ris_entry(record) for record in RECORDS) + "\n", encoding="utf-8")

    doi_count = sum(1 for record in RECORDS if record.doi)
    eswa_count = sum(1 for record in RECORDS if record.venue == "Expert Systems with Applications")
    no_doi_records = [record for record in RECORDS if not record.doi]
    lines = [
        "# 初稿6 最终34篇文献 Zotero 导入说明",
        "",
        f"- BibTeX 文件：{BIB_PATH}",
        f"- RIS 文件：{RIS_PATH}",
        f"- 条目总数：{len(RECORDS)}",
        f"- DOI 条目数：{doi_count}",
        f"- 无 DOI 条目数：{len(no_doi_records)}",
        f"- Expert Systems with Applications 条目数：{eswa_count}",
        "",
        "## 无 DOI 条目",
        "",
    ]
    if no_doi_records:
        for record in no_doi_records:
            lines.append(f"- {record.title}，保留 URL：{record.url}")
    else:
        lines.append("- 无")
    lines.extend(
        [
            "",
            "## 导入建议",
            "",
            "建议优先导入 RIS 文件；如果 Zotero 中已有部分文献，导入后可以使用 Zotero 的重复条目合并功能进行去重。BibTeX 文件作为备用，也适合后续 LaTeX 或 Overleaf 使用。",
        ]
    )
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with ZipFile(ZIP_PATH, "w", compression=ZIP_DEFLATED) as package:
        package.write(BIB_PATH, BIB_PATH.name)
        package.write(RIS_PATH, RIS_PATH.name)
        package.write(SUMMARY_PATH, SUMMARY_PATH.name)


def main() -> None:
    write_outputs()
    print(f"已生成 BibTeX：{BIB_PATH}")
    print(f"已生成 RIS：{RIS_PATH}")
    print(f"已生成说明：{SUMMARY_PATH}")
    print(f"已生成压缩包：{ZIP_PATH}")
    print(f"条目总数：{len(RECORDS)}")
    print(f"DOI 条目数：{sum(1 for record in RECORDS if record.doi)}")
    print(f"无 DOI 条目数：{sum(1 for record in RECORDS if not record.doi)}")


if __name__ == "__main__":
    main()
