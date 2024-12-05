import requests
import tiktoken
import instructor
from pydantic import ValidationError
from openai import OpenAI
import time

enc = tiktoken.get_encoding("cl100k_base")
assert enc.decode(enc.encode("hello world")) == "hello world"


def LLaMA_response_json(
    messages, model_name, response_model, url="http://localhost:11434/v1"
):
    """
    LLM module to be called

    Args
        messages: list of message dictionaries following ChatCompletion format
        model_name: name of the LLaMA model
        url: endpoint where LLaMA is hosted
    """
    MAX_RETRY = 7
    count = 0
    try:
        # enables `response_model` in create call
        client = instructor.from_openai(
            OpenAI(
                base_url=url,
                api_key="ollama",  # required, but unused
            ),
            mode=instructor.Mode.JSON,
        )
        while True:
            if count >= MAX_RETRY:
                print(
                    f"""Max retries reach. LLM failed at outputing the correct result. 
                Input parameter response_model might have issue: {response_model.schema_json(indent=2)}"""
                )
                raise ValidationError
            try:
                # Use instructor to handle the structured response
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    response_model=response_model,  # Use the Character model to structure the response
                )
                # Print the structured output as JSON
                response = response.model_dump_json()
                token_num_count = sum(
                    len(enc.encode(msg["content"])) for msg in messages
                ) + len(enc.encode(response))
                return response, token_num_count
            except Exception as e:
                count += 1
                print(
                    f"Validation failed during LLM output generation: {e} for {count} times. Retrying..."
                )
    except Exception as e:
        print(f"API call failed: {e}")
        return None, 0

def GPT_response(messages, model_name):
    token_num_count = 0
    client = OpenAI( # Enter Your API Key Here
        api_key=''
    )
    # for item in messages:
    #     token_num_count += len(enc.encode(item["content"]))

    try:
        time.sleep(5)
        result = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.0,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
    except Exception as e:
        print(e)
        try:
            time.sleep(5)
            result = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.0,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
            )
        except:
            try:
                print(f'{model_name} Waiting 60 seconds for API query')
                time.sleep(60)
                result = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature = 0.0,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0
                )
            except:
                return 'Out of tokens', token_num_count
    #token_num_count += len(enc.encode(result.choices[0].message.content))
    return result.choices[0].message.content, token_num_count

def LLaMA_response(messages, model_name, url="http://localhost:11434/api/generate"):
    """
    LLM module to be called

    Args
        messages: list of message dictionaries following ChatCompletion format
        model_name: name of the LLaMA model
        url: endpoint where LLaMA is hosted
    """
    prompt = "\n".join(
        [f"{msg['role'].capitalize()}: {msg['content']}" for msg in messages]
    )
    data = {
        "model": model_name,
        "prompt": prompt,
        "temperature": 0.0,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "stream": False,
    }

    try:
        response = requests.post(url=url, json=data)

        if response.status_code == 200:
            response_text = response.json().get("response", "")
            token_num_count = sum(
                len(enc.encode(msg["content"])) for msg in messages
            ) + len(enc.encode(response_text))
            # print(f"Token_num_count: {token_num_count}")
            return response_text, token_num_count
        else:
            print("Error:", response.status_code, response.json())
            return None, 0
    except Exception as e:
        print(f"API call failed: {e}")
        return None, 0