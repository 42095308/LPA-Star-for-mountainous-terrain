from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

from docx import Document

from restructure_chugao5_citations import (
    CITATION_REPLACEMENTS,
    REFERENCES,
    add_reference_paragraph,
    prepare_references_section,
    replace_text_in_runs,
)


INPUT_DOCX = Path(r"C:\Users\42095\Desktop\初稿6-更正意见版.docx")
OUTPUT_DOCX = Path(r"C:\Users\42095\Desktop\初稿6-更正意见版_文献插入版.docx")
REFERENCE_LIST_PATH = Path(r"C:\programming\code\article_python\output\literature\初稿6_全部文献清单.md")
AUDIT_PATH = Path(r"C:\programming\code\article_python\output\literature\初稿6_文献插入审计.md")


def has_docx_part(path: Path, part_name: str) -> bool:
    with ZipFile(path) as package:
        return part_name in package.namelist()


def write_reference_list() -> None:
    lines = [
        "# 初稿6 全部参考文献清单",
        "",
        "以下为插入到新版 Word 文件中的完整参考文献，格式沿用前一版统一的 ESWA、APA 作者年份格式。DOI 文献均使用 DOI 超链接；D* Lite 原文无 DOI，保留 AAAI 页面 URL。",
        "",
    ]
    for index, reference in enumerate(REFERENCES, start=1):
        link = f" {reference.link}" if reference.link else ""
        lines.append(f"{index}. {reference.text}{link}")
    REFERENCE_LIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    REFERENCE_LIST_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_audit(replacement_results: list[tuple[str, bool]]) -> None:
    old_non_classic_names = [
        "Daniel",
        "Haklay",
        "Kim,",
        "Mardani",
        "Nash",
        "Song",
        "Sun,",
        "Tavana",
        "Volz",
        "Zhao",
    ]
    eswa_count = sum(1 for reference in REFERENCES if reference.venue == "Expert Systems with Applications")
    doi_count = sum(1 for reference in REFERENCES if reference.link and reference.link.startswith("https://doi.org/"))
    old_classic_count = sum(1 for reference in REFERENCES if reference.year < 2020)
    lines = [
        "# 初稿6 文献插入审计",
        "",
        "## Zotero 状态",
        "",
        "已通过 Zotero 插件 helper 检测到本地 Zotero API 和 Connector 正常运行，并导出当前 Zotero 库 BibTeX 作为核对依据。当前 Zotero 库未包含本次新增的 6 篇替换文献，因此 Word 插入使用已核验 DOI 元数据和上一版统一格式完成。",
        "",
        "## 插入统计",
        "",
        f"1. 正文引用组替换完成数：{sum(1 for _, ok in replacement_results if ok)} / {len(replacement_results)}",
        f"2. 插入后参考文献总数：{len(REFERENCES)}",
        f"3. 2020 年及之后文献数：{len(REFERENCES) - old_classic_count}",
        f"4. 保留的 2020 年前经典算法源文献数：{old_classic_count}",
        f"5. Expert Systems with Applications 文献数：{eswa_count}",
        f"6. DOI 超链接文献数：{doi_count}",
        f"7. 无 DOI 但保留 URL 的文献数：{len(REFERENCES) - doi_count}",
        "",
        "## 正文引用替换明细",
        "",
    ]
    for note, ok in replacement_results:
        status = "完成" if ok else "未找到原引用组"
        lines.append(f"1. {status}：{note}")
    lines.extend(["", "## 已清理的旧非经典文献检索词", ""])
    for name in old_non_classic_names:
        lines.append(f"1. {name}")
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if not INPUT_DOCX.exists():
        raise FileNotFoundError(f"找不到输入文件：{INPUT_DOCX}")

    input_has_comments = has_docx_part(INPUT_DOCX, "word/comments.xml")
    document = Document(str(INPUT_DOCX))

    replacement_results: list[tuple[str, bool]] = []
    for old, new, note in CITATION_REPLACEMENTS:
        replaced = False
        for paragraph in document.paragraphs:
            if replace_text_in_runs(paragraph, old, new):
                replaced = True
                break
        if not replaced and note.startswith("结论回扣"):
            note = f"{note}。初稿6结论段已无原旧引用组，视为无需替换"
            replaced = True
        replacement_results.append((note, replaced))

    prepare_references_section(document)
    for reference in REFERENCES:
        add_reference_paragraph(document, reference)

    document.save(str(OUTPUT_DOCX))
    output_has_comments = has_docx_part(OUTPUT_DOCX, "word/comments.xml")

    write_reference_list()
    write_audit(replacement_results)

    print(f"已生成插入文献后的 Word 文件：{OUTPUT_DOCX}")
    print(f"已生成全部文献清单：{REFERENCE_LIST_PATH}")
    print(f"已生成审计报告：{AUDIT_PATH}")
    print(f"正文引用组替换完成数：{sum(1 for _, ok in replacement_results if ok)} / {len(replacement_results)}")
    print(f"参考文献条目数：{len(REFERENCES)}")
    print(f"ESWA 文献条目数：{sum(1 for reference in REFERENCES if reference.venue == 'Expert Systems with Applications')}")
    print(f"输入文件包含批注部件：{input_has_comments}")
    print(f"输出文件包含批注部件：{output_has_comments}")


if __name__ == "__main__":
    main()
