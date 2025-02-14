from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
from crewai.knowledge.source.json_knowledge_source import JSONKnowledgeSource
from .models import (
    JobRequirements,
    ResumeOptimization,
    CompanyResearch
)


@CrewBase
class ResumeCrew():
    """ResumeCrew for building a new ATS-optimized CV from scratch based on a job spec and interview preparation."""

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'
    def __init__(self) -> None:
        """
        Initialize with candidate data from a resume.json file.
        The JSON follows the expected schema including personal information,
        work experience, education, skills, and projects.
        """
        self.resume_json = JSONKnowledgeSource(file_paths="resume.json")
    
    @agent
    def job_analyzer(self) -> Agent:
        return Agent(
            config=self.agents_config['job_analyzer'],
            verbose=True,
            tools=[ScrapeWebsiteTool()],
            llm=LLM("gpt-4o")
        )

    @agent
    def company_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['company_researcher'],
            verbose=True,
            tools=[SerperDevTool()],
            llm=LLM("gpt-4o"),
            knowledge_sources=[self.resume_json]
        )

    @agent
    def resume_writer(self) -> Agent:
        """
        Write an entirely new CV in markdown format from scratch.
        The generated CV will explicitly reference the job specification
        """
        return Agent(
            config=self.agents_config['resume_writer'],
            verbose=True,
            llm=LLM("gpt-4o"),
            knowledge_sources=[self.resume_json]
        )

    @agent
    def resume_analyzer(self) -> Agent:
        """
        Analyze the candidate’s draft resume
        """
        return Agent(
            config=self.agents_config['resume_analyzer'],
            verbose=True,
            llm=LLM("gpt-4o")
        )

    @agent
    def report_generator(self) -> Agent:
        return Agent(
            config=self.agents_config['report_generator'],
            verbose=True,
            llm=LLM("gpt-4o")
        )

    @task
    def analyze_job_task(self) -> Task:
        return Task(
            config=self.tasks_config['analyze_job_task'],
            output_file='output/job_analysis.json',
            output_pydantic=JobRequirements
        )

    @task
    def research_company_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_company_task'],
            output_file='output/company_research.json',  
            output_pydantic=CompanyResearch
        )

    @task
    def generate_resume_task(self) -> Task:
        return Task(
            config=self.tasks_config['generate_resume_task'],
            output_file='output/draft_resume.md'
        )

    @task
    def optimize_resume_task(self) -> Task:
        return Task(
            config=self.tasks_config['optimize_resume_task'],
            output_file='output/final_resume.json'
        )

    @task
    def generate_report_task(self) -> Task:
        return Task(
            config=self.tasks_config['generate_report_task'],
            output_file='output/final_report.md'
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            verbose=True,
            process=Process.sequential,
            knowledge_sources=[self.resume_json]
        )
