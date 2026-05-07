from operator import add
from pathlib import Path
from typing import TypedDict, Optional, List, Annotated

from langgraph.constants import START, END
from langgraph.graph import StateGraph


# The structure of the logs
class Log(TypedDict):
    id: str
    question: str
    docs: Optional[List]
    answer: str
    grade: Optional[int]
    grader: Optional[str]
    feedback: Optional[str]


 # Define subgraph to summarize logs and find failure cases

# Failure Analysis Sub-graph
class FailureAnalysisState(TypedDict):
    cleaned_logs: List[Log]
    failures: List[Log]
    fa_summary: str
    processed_logs: List[str]

class FailureAnalysisOutputState(TypedDict):
    fa_summary: str
    processed_logs: List[str]

def get_failures(state):
    """ Get logs that contain a failure """
    cleaned_logs = state["cleaned_logs"]
    failures = [log for log in cleaned_logs if "grade" in log]
    return {"failures": failures}

def generate_summary(state):
    """ Generate summary of failures """
    failures = state["failures"]
    # Add fxn: fa_summary = summarize(failures)
    fa_summary = "Poor quality retrieval of Chroma documentation."
    return {"fa_summary": fa_summary, "processed_logs": [f"failure-analysis-on-log-{failure['id']}" for failure in failures]}


def failure_analysis_subgraph():
    fa_builder = StateGraph(state_schema=FailureAnalysisState, output_schema=FailureAnalysisOutputState)
    fa_builder.add_node("get_failures", get_failures)
    fa_builder.add_node("generate_summary", generate_summary)
    fa_builder.add_edge(START, "get_failures")
    fa_builder.add_edge("get_failures", "generate_summary")
    fa_builder.add_edge("generate_summary", END)

    graph = fa_builder.compile()

    graph.get_graph(xray=1).print_ascii()
    graph.get_graph(xray=1).draw_mermaid_png(
    output_file_path="project_langgraph/graph_images/failure_analysis_subgraph.png")

    return graph
#--------------------------------------------------------------------------
# Summarization subgraph
class QuestionSummarizationState(TypedDict):
    cleaned_logs: List[Log]
    qs_summary: str
    report: str
    processed_logs: List[str]

class QuestionSummarizationOutputState(TypedDict):
    report: str
    processed_logs: List[str]

def generate_summary(state):
    cleaned_logs = state["cleaned_logs"]
    # Add fxn: summary = summarize(generate_summary)
    summary = "Questions focused on usage of ChatOllama and Chroma vector store."
    return {"qs_summary": summary, "processed_logs": [f"summary-on-log-{log['id']}" for log in cleaned_logs]}

def send_to_slack(state):
    qs_summary = state["qs_summary"]
    # Add fxn: report = report_generation(qs_summary)
    report = "foo bar baz"
    return {"report": report}

def question_summarization_subgraph():
    qs_builder = StateGraph(QuestionSummarizationState, output_schema=QuestionSummarizationOutputState)
    qs_builder.add_node("generate_summary", generate_summary)
    qs_builder.add_node("send_to_slack", send_to_slack)
    qs_builder.add_edge(START, "generate_summary")
    qs_builder.add_edge("generate_summary", "send_to_slack")
    qs_builder.add_edge("send_to_slack", END)

    graph = qs_builder.compile()

    graph.get_graph(xray=1).print_ascii()
    graph.get_graph(xray=1, ).draw_mermaid_png(
        output_file_path="project_langgraph/graph_images/question_summarization_subgraph.png")
    return graph
#------------------------------------------------------------------------

# Entry Graph
class EntryGraphState(TypedDict):
    raw_logs: List[Log]
    cleaned_logs: List[Log]
    fa_summary: str # This will only be generated in the FA sub-graph
    report: str # This will only be generated in the QS sub-graph
    processed_logs:  Annotated[List[int], add] # This will be generated in BOTH sub-graphs

def clean_logs(state):
    # Get logs
    raw_logs = state["raw_logs"]
    # Data cleaning raw_logs -> docs
    cleaned_logs = raw_logs
    return {"cleaned_logs": cleaned_logs}

def entry_graph():
    entry_builder = StateGraph(EntryGraphState)
    entry_builder.add_node("clean_logs", clean_logs)
    entry_builder.add_node("question_summarization", question_summarization_subgraph())
    entry_builder.add_node("failure_analysis", failure_analysis_subgraph())

    entry_builder.add_edge(START, "clean_logs")
    entry_builder.add_edge("clean_logs", "failure_analysis")
    entry_builder.add_edge("clean_logs", "question_summarization")
    entry_builder.add_edge("failure_analysis", END)
    entry_builder.add_edge("question_summarization", END)

    graph = entry_builder.compile()

    graph.get_graph(xray=1).print_ascii()
    graph.get_graph(xray=1).draw_mermaid_png(
        output_file_path="project_langgraph/graph_images/subgraph.png")
    return graph

# Dummy logs
question_answer = Log(
    id="1",
    question="How can I import ChatOllama?",
    answer="To import ChatOllama, use: 'from langchain_community.chat_models import ChatOllama.'",
)

question_answer_feedback = Log(
    id="2",
    question="How can I use Chroma vector store?",
    answer="To use Chroma, define: rag_chain = create_retrieval_chain(retriever, question_answer_chain).",
    grade=0,
    grader="Document Relevance Recall",
    feedback="The retrieved documents discuss vector stores in general, but not Chroma specifically",
)



def main():
    print("\n\n-----------------failure_analysis_subgraph --------------------------\n\n")
    failure_subgraph = failure_analysis_subgraph()
    print("\n\n-----------------summarization_subgraph --------------------------\n\n")
    summarization_subgraph = question_summarization_subgraph()

    print("\n\n-----------------main_graph --------------------------\n\n")
    main_graph = entry_graph()

    print("\n\n-----------------Graph execution  --------------------------\n\n")
    raw_logs = [question_answer, question_answer_feedback]
    res = main_graph.invoke({"raw_logs": raw_logs})
    print("\n\n-----------------final output --------------------------\n\n")
    print(f"Full response \n {res} ")
    print("-------------")
    print(f"raw_logs=={ res["raw_logs"]}")
    print(f"report=={ res["report"]}")
    print(f"processed_logs=={ res["processed_logs"]}")
    print(f"cleaned_logs=={res["cleaned_logs"]}")

    # raw_logs==[{'id': '1', 'question': 'How can I import ChatOllama?', 'answer': "To import ChatOllama, use: 'from langchain_community.chat_models import ChatOllama.'"}, {'id': '2', 'question': 'How can I use Chroma vector store?', 'answer': 'To use Chroma, define: rag_chain = create_retrieval_chain(retriever, question_answer_chain).', 'grade': 0, 'grader': 'Document Relevance Recall', 'feedback': 'The retrieved documents discuss vector stores in general, but not Chroma specifically'}]
    # report==foo bar baz
    # processed_logs==['summary-on-log-1', 'summary-on-log-2', 'summary-on-log-1', 'summary-on-log-2']
    # cleaned_logs==[{'id': '1', 'question': 'How can I import ChatOllama?', 'answer': "To import ChatOllama, use: 'from langchain_community.chat_models import ChatOllama.'"}, {'id': '2', 'question': 'How can I use Chroma vector store?', 'answer': 'To use Chroma, define: rag_chain = create_retrieval_chain(retriever, question_answer_chain).', 'grade': 0, 'grader': 'Document Relevance Recall', 'feedback': 'The retrieved documents discuss vector stores in general, but not Chroma specifically'}]




# uv run -m project_langgraph.sub_graph_summarize_logs_and_find_failure
if __name__ == '__main__':
    main()