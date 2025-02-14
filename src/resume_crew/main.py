#!/usr/bin/env python
import sys
import warnings

from resume_crew.crew import ResumeCrew

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

def run():
    """
    Run the resume optimization crew. 
    """
    inputs = {
        'job_url': 'https://www.titansoft.com/en/career/39',
        'company_name': 'Titansoft'
    }
    ResumeCrew().crew().kickoff(inputs=inputs)

if __name__ == "__main__":
    run()
