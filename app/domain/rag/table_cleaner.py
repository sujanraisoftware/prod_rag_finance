import re


TABLE_ROW_REGEX = re.compile(r"^\s*\|.*\|\s*$")
TABLE_DELIMITER_REGEX = re.compile(r"^\s*\|(?:\s*:?-+:?\s*\|)+\s*$")
FINANCIAL_NUMBER_REGEX = re.compile(r"[\$€£¥]?\s*\(?[\d,]+(?:\.\d+)?\)?%?")


def is_markdown_table(text: str) -> bool:
    """
    Checks if a given text block contains a valid markdown table structure.
    Must contain at least one delimiter line |---| and multiple rows.
    """
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if len(lines) < 2:
        return False

    has_delimiter = any(TABLE_DELIMITER_REGEX.match(line) for line in lines)
    if not has_delimiter:
        return False

    table_rows = sum(1 for line in lines if TABLE_ROW_REGEX.match(line))
    return table_rows >= 2


def clean_table_markdown(table_text: str) -> str:
    """
    Cleans and standardizes markdown tables extracted from financial PDFs:
    - Normalizes cell spacing
    - Converts messy line breaks in cells (<br>, <br/>)
    - Strips empty leading/trailing pipes without breaking column alignment
    """
    lines = table_text.strip().split("\n")
    cleaned_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        if stripped.startswith("|") and stripped.endswith("|"):
            # Replace internal <br> with space or comma if it splits numeric values
            cleaned_cell_line = re.sub(r"<br\s*/?>", " ", stripped)
            # Normalize excessive spaces between pipes
            cleaned_lines.append(cleaned_cell_line)
        else:
            cleaned_lines.append(stripped)

    return "\n".join(cleaned_lines)


def slice_large_table(
    table_text: str,
    max_rows_per_slice: int = 40,
    context_prefix: str = ""
) -> list[str]:
    """
    Splits an oversized financial table across multiple chunks WITHOUT losing column context.
    Repeats the header row and delimiter row on each sliced table chunk.
    """
    lines = [line.strip() for line in table_text.strip().split("\n") if line.strip()]
    if len(lines) <= 2:
        return [table_text]

    # Find delimiter line index
    delimiter_idx = -1
    for i, line in enumerate(lines):
        if TABLE_DELIMITER_REGEX.match(line):
            delimiter_idx = i
            break

    if delimiter_idx == -1 or delimiter_idx >= len(lines) - 1:
        # Fallback if no clear header delimiter
        return [table_text]

    header_block = lines[: delimiter_idx + 1]
    body_rows = lines[delimiter_idx + 1 :]

    if len(body_rows) <= max_rows_per_slice:
        return [table_text]

    slices = []
    total_slices = (len(body_rows) + max_rows_per_slice - 1) // max_rows_per_slice

    for slice_num in range(total_slices):
        start_idx = slice_num * max_rows_per_slice
        end_idx = min(start_idx + max_rows_per_slice, len(body_rows))
        current_rows = body_rows[start_idx:end_idx]

        slice_header = list(header_block)
        annotation = f"*Table part {slice_num + 1} of {total_slices} (continued)*\n"
        if context_prefix:
            annotation = f"*{context_prefix} (Part {slice_num + 1} of {total_slices})*\n"

        sliced_table = annotation + "\n".join(slice_header + current_rows)
        slices.append(sliced_table)

    return slices

