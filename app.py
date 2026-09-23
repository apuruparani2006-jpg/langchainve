import os
import tempfile
import traceback
import subprocess

from typing import TypedDict, List, Optional

from flask import Flask, render_template, request
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI


app = Flask(__name__)

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    print("WARNING: GEMINI_API_KEY environment variable not found.")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    google_api_key=API_KEY,
    temperature=0
)


class CrewState(TypedDict):
    messages: List[BaseMessage]
    next_step: Optional[str]
    code: Optional[str]
    testbench: Optional[str]
    report: Optional[str]


def clean_code(text):
    if not isinstance(text, str):
        text = str(text)

    return (
        text.replace("```verilog", "")
        .replace("```systemverilog", "")
        .replace("```", "")
        .strip()
    )


def get_response_text(response):
    content = response.content

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        result = ""
        for item in content:
            if isinstance(item, dict):
                result += item.get("text", "")
            else:
                result += str(item)
        return result

    return str(content)


@tool
def run_verilog_code(code: str) -> str:
    """Compile Verilog/SystemVerilog using Icarus Verilog."""

    code = clean_code(code)

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            verilog_file = os.path.join(temp_dir, "design.v")
            output_file = os.path.join(temp_dir, "design.out")

            with open(verilog_file, "w", encoding="utf-8") as f:
                f.write(code)

            result = subprocess.run(
                [
                    "iverilog",
                    "-g2012",
                    "-o",
                    output_file,
                    verilog_file
                ],
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                return (
                    "VERILOG COMPILATION PASSED\n\n"
                    "No syntax errors were detected."
                )

            return (
                "VERILOG COMPILATION FAILED\n\n"
                + result.stderr
            )

    except FileNotFoundError:
        return (
            "Icarus Verilog is not installed. "
            "Please install Icarus Verilog."
        )

    except Exception:
        return "Compilation Error:\n" + traceback.format_exc()


@tool
def generate_test_cases(task_description: str) -> str:
    """Generate verification scenarios for a Verilog task."""

    prompt = f"""
You are a Senior RTL Verification Engineer.

Analyze this Verilog design task:

{task_description}

Generate 5 specific verification scenarios.

Include:
1. Normal operation
2. Boundary condition
3. Edge case
4. Reset behavior if applicable
5. Clock/timing behavior if applicable

Return only a numbered list.
Do not generate code.
"""

    response = llm.invoke(prompt)
    return get_response_text(response)


def developer_node(state: CrewState):
    task = state["messages"][-1].content

    prompt = f"""
You are an expert Verilog and RTL design engineer.

Create a clean, correct and synthesizable
Verilog/SystemVerilog design for:

{task}

Rules:
1. Return ONLY Verilog/SystemVerilog code.
2. Do not provide explanations.
3. Do not use Markdown.
4. Include a proper module declaration.
5. Use meaningful signal names.
6. Use synthesizable RTL.
7. Include clock and reset only when required.
8. Make the design compatible with Icarus Verilog.
9. Do not create a testbench.
10. Do not include ``` markers.
"""

    response = llm.invoke(prompt)
    code = clean_code(get_response_text(response))

    return {"code": code}


def tester_node(state: CrewState):
    task = state["messages"][-1].content

    test_cases = generate_test_cases.invoke(task)

    compilation_result = run_verilog_code.invoke(
        {"code": state["code"]}
    )

    report = f"""
VERILOG COMPILATION RESULT
==========================

{compilation_result}


TEST SCENARIOS
==============

{test_cases}
"""

    return {"report": report}


def manager_node(state: CrewState):
    return {"next_step": "done"}


workflow = StateGraph(CrewState)

workflow.add_node("developer", developer_node)
workflow.add_node("tester", tester_node)
workflow.add_node("manager", manager_node)

workflow.add_edge(START, "developer")
workflow.add_edge("developer", "tester")
workflow.add_edge("tester", "manager")
workflow.add_edge("manager", END)

verilog_app = workflow.compile()


@app.route("/", methods=["GET", "POST"])
def index():
    code = ""
    report = ""
    task = ""

    if request.method == "POST":
        task = request.form.get("task", "").strip()

        if task:
            initial_state = {
                "messages": [HumanMessage(content=task)],
                "next_step": None,
                "code": None,
                "testbench": None,
                "report": None
            }

            result = verilog_app.invoke(initial_state)

            code = result.get("code", "")
            report = result.get("report", "")

    return render_template(
        "index.html",
        task=task,
        code=code,
        report=report
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", 5000)),
        debug=True
    )
