# RAG

## PREPARATION

```
from langchain import chunker
from openai import calculate_embeddings, llm
from pinecone import vectordb


file = "`document.pdf"

chunks = chunker(file)
print(chunks)

[para1, para2, ...]

embeddings_store = {}
for chunk in chunks:
    embedding = calculate_embeddings(chunk)
    embeddings_store.update({embedding : chunk})


vectordb.store(embeddings_store)
```



## RUNTIME: WHEN ANSWERING A QUESTION

```
question = "when is her bday"

question_embedding = calculate_embeddings(question)

best_matching_chunks = vectordb.search(question_embedding, top_n= 10, threshold=0.6)

print(best_matching_chunks)

[para1, para8, ...]

prompt = f"CONTEXT: {best_matching_chunks} \n QUESTION: {question}"

answer = llm(prompt)

print(answer)
```



# DIFFERENT WAYS TO ANSWER QUESTIONS

```
propmt = "you have to find every song which is about birds"

llm(propmt)

[song1, song2, ...]
```
Suppose the LLM responds with 10 songs.

Why do you think that is? It only 10 songs probably because there's only 10 songs in the training data.

## HOW TO ANSWER THE QUESTION IF THE INFO IS NOT IN THE LLM TRAINING DATA?

There are a few methods:

1. PRE-TRAINING

train it on more songs (expensive)

2. IN-CONTEXT LEARNING

give it the lyrics of all songs in the world in propmt / context (thats too much text, wont fit)

3. RAG

we store all song lyrics in vectordb (along with titles) and search for the  question "bird".
then we get songs about birds.
then we give this list of songs to an llm. ask it "filter the songs which are 100% acutally anbout birds."
then the llm will give us the result.

4. FINE-TUNING

fine-tune the llm on song lyrics data.






# TOOL USE

Let's create a system which can take a user's request, figure out which tools to use in order to do the task, and successfully use those tools to finish the task!

the system can handle 2 tasks.

```

task_1 = "send an invite to sarah for 10 am coffee tomorrow"
task_2 = "send an email to sarah@gmail.com about her bad work ethic"


def send_calendar_invite(name, time, date, purpose):
    pass

def send_email(email_address, subject, body):
    pass



instructions = """here are a few tools.

you will figure out whats the required tool for the user's request.

and, you will give the parameters to use that tool as well.


tool 1. send_email

{"email_address" : "Email address here",
"subject" : "Email subject here",
"body" : "Email body here"
}


tool 2. send_calendar_invite

{"name" : "Person name here",
"time" : "Time for invite here",
"date" : "Date for invite here",
"purpose": "Purpose of meeting here"
}

Make sure you respond ONLY in the required format or else it will not be parsed correctly!!!
"""


## TASK 1

llm(instructions + task_1)

tool = "send_calendar_invite"
data = {"name" : "Sarah",
 "time" : "10 am",
 "date" : "10 Jan",
 "purpose" : "Coffee"}

if tool == "send_calendar_invite":
    send_calendar_invite(data["name"], data["time"],data["date"],data["purpose"])
elif tool == "send_email":
    send_email(data["email_adress"], data["subject"], data["body"])

## TASK 2

llm(instructions + task_2)
tool = send_email
data = {"email_adress" : "sarah@gmail.com",
        "subject" : "bad work ethic",
        "body" : "you suck"
        }

if tool == "send_calendar_invite":
    send_calendar_invite(data["name"], data["time"],data["date"],data["purpose"])
elif tool == "send_email":
    send_email(data["email_adress"], data["subject"], data["body"])
    
```
