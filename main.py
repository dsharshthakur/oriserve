# import required libraries
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate


# class responsible for using gemini-pro to generate response
class UserSentiment:
    # initialization
    def __init__(self, api_key, llm_name):
        self.llm_api_key = api_key
        self.llm_name = llm_name

    # function responsible for generating response
    def generate(self, user_review):
        # LLM - (Google)
        llm = GoogleGenerativeAI(model=self.llm_name, google_api_key=self.llm_api_key, model_kwrgs={"temperature": 0.2})

        template = ''' 
        You are a great AI assistant which can easily perform "subtheme sentiments analysis" from a given  user review. 
        The text is {text} . Now, Identify all the subtheme sentiments from this text with highest accuracy, and don't 
        miss any insight from the text also avoid giving irrelevant sentiments in output.
        Your answer should be strictly according to the given structure only.

        For example 1:
        text: It was very straightforward and the garage was great. Hadn't even known about them before.
        answer: [garage service positive]
        
        For example 2:
        text: Easy to use, also good price.
        answer should be: [value for money positive]
        
        For example 3:
        text: Staff with [REDACTED] went above and beyond to help me organise my booking and ordering my tires. The staff within the chosen garage were also quick and efficient. :).
        answer should be: [ease of booking positive, length of fitting positive, advisor/agent service positive] 
        
        For example 4:
        text: Simple to order. Easy to book your appointment. Tyres turned up on time. Fitters did a great job.
        answer should be:[delivery punctuality positive, ease of booking positive]
        
        For example 5:
        text:Not a 10 as I found the garage a bit difficult to find
        answer should be: [location negative]
    
        For example 6:
        text:Good polite and professional service from Richard Beavers in Minehead.
        Bit of confusion intitially over when the tyres could be fitted but everything worked out in the end.
        answer should be:[booking confusion negative, garage service positive]
        
        For example 7:
        text: Great service, they had a supply issue with my original order and so upgraded to the next tyre up without charging the extra cost.
        Also they saw what could be an issue and highlighted it so that i could double check i had the correct order.
        answer: [extra charges positive, no stock negative, advisor/agent service positive]
        '''
        # prompt template for the llm - Instruction for the model
        prompt = PromptTemplate(template=template, input_variables=["text"])

        # combining the prompt and the model
        chain = prompt | llm

        # ai response
        ai_response = chain.invoke({'text': user_review})

        return ai_response
