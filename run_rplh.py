from LLM import *
from prompt import *
from env_create import *
import os
import json
import re
import sys
import os

def run_exp(Saving_path,
            pg_row_num,
            pg_column_num,
            iteration_num,
            query_time_limit,
            dialogue_history_method = '_w_only_state_action_history'
            ):
  
  '''This is information constant'''
  data_dict = {
        'user_prompt_list': [],
        'response_total_list': [],
        'pg_state_list': [],
        'dialogue_history_list': [],
        'token_num_count_list': [],
        'hca_agent_response_list': [],
        'pg_dict': None  # For initial environment state
    }
    
  # Load initial environment state
  with open(Saving_path+f'/env_pg_state_{pg_row_num}_{pg_column_num}/pg_state{iteration_num}/pg_state{iteration_num}.json', 'r') as file:
    pg_dict = json.load(file)
    data_dict['pg_dict'] = pg_dict

  num_agent = pg_row_num + pg_column_num
  data_dict['pg_state_list'].append(data_dict['pg_dict'])

  print(f'query_time_limit: {query_time_limit}')
  for index_query_times in range(2):
    state_update_prompt = state_update_func(pg_row_num, pg_column_num,  data_dict['pg_dict'])

    #-----------------------------------------ONE AGENT THINK BY THEMSELVES ONCE-----------------------------------------#
    for a in range(num_agent):
      print(f'-------###-------###-------###-------HCA_AGENT_{a}-------###-------###-------###-------')

      '''FOR NUM_AGENT, ITERATIVELY DO'''
      user_prompt_1 = rplh_prompt_func(state_update_prompt, data_dict, dialogue_history_method)
      data_dict['user_prompt_list'].append(user_prompt_1)
      messages = message_construct_func([user_prompt_1], [], dialogue_history_method)
      response, token_num_count = LLaMA_response(messages, model_name)
      print(response)
              
      #-----------------------------------------SYNTACTIC CHECK-----------------------------------------#
      data_dict['token_num_count_list'].append(token_num_count)
      match = re.search(r'{.*}', response, re.DOTALL)
      if match:
        response = match.group()
      if response[0] == '{' and response[-1] == '}':
        response, token_num_count_list_add = with_action_syntactic_check_func(data_dict['pg_dict'],
                                                                              response,
                                                                              [user_prompt_1],
                                                                              [],
                                                                              model_name,
                                                                              '_w_all_dialogue_history',
                                                                              )
        data_dict['token_num_count_list'] = data_dict['token_num_count_list'] + token_num_count_list_add
        print(f'AGENT ACTION RESPONSE: {response}')
      else:
        raise ValueError(f'Response format error: {response}')
      if response == 'Out of tokens':
        pass
      elif response == 'Syntactic Error':
        pass
      data_dict['hca_agent_response_list'].append(response)
      '''This for loop ends here for all agents doing centralized planning by themselves'''

      #-----------------------------------------FOR EACH AGENT RECIEVES COMMAND FROM THE CURRENT HELLUCINATING MAIN AGENT-----------------------------------------#
      dialogue_history = ''
      print(f'ORIGINAL PLAN: {response}')
      data_local={
        'prompt_list_dir': {},
        'response_list_dir': {},
        'local_agent_response_list_dir': {},
        'agent_dict': {}
      }

      data_local['local_agent_response_list_dir']['feedback1'] = ''
      data_local['agent_dict'] = json.loads(response)

      for local_agent_row_i in range(pg_row_num):
        for local_agent_column_j in range(pg_column_num):
          print(f'-------###-------###-------###-------LOCAL_ROW_{local_agent_row_i}_COL_{local_agent_column_j}-------###-------###-------###-------')

          if f'Agent[{local_agent_row_i+0.5}, {local_agent_column_j+0.5}]' in data_local['agent_dict']:
            # note, dict, this have space
            data_local['prompt_list_dir'][f'Agent[{local_agent_row_i+0.5}, {local_agent_column_j+0.5}]'] = []
            data_local['response_list_dir'][f'Agent[{local_agent_row_i+0.5}, {local_agent_column_j+0.5}]'] = []
            print(data_local)
            
            state_update_prompt_local_agent, state_update_prompt_other_agent =  state_update_func_local_agent(pg_row_num,
                                                                                                              pg_column_num,
                                                                                                              local_agent_row_i,
                                                                                                              local_agent_column_j,
                                                                                                              data_dict['pg_dict'])
            local_reprompt = dialogue_func(state_update_prompt_local_agent,
                                            state_update_prompt_other_agent,
                                            response,
                                            data_dict,
                                            dialogue_history_method
                                            )
            data_local['prompt_list_dir'][f'Agent[{local_agent_row_i+0.5}, {local_agent_column_j+0.5}]'].append(local_reprompt)
            message_raw = data_local['prompt_list_dir'][f'Agent[{local_agent_row_i+0.5}, {local_agent_column_j+0.5}]']
            response_raw = data_local['response_list_dir'][f'Agent[{local_agent_row_i+0.5}, {local_agent_column_j+0.5}]']

            messages = message_construct_func(message_raw,
                                              response_raw,
                                              '_w_all_dialogue_history')
            response_local_agent, token_num_count = LLaMA_response(messages, model_name)
            data_dict['token_num_count_list'].append(token_num_count)
            
            if response_local_agent != 'I Agree':
              data_local['local_agent_response_list_dir']['feedback1'] += f'Agent[{local_agent_row_i+0.5}, {local_agent_column_j+0.5}]: {response_local_agent}\n'
              dialogue_history += f'Agent[{local_agent_row_i+0.5}, {local_agent_column_j+0.5}]: {response_local_agent}\n'
          
          #-----------------------------------------RECONSTRUCT MESSAGES-----------------------------------------#
          if data_local['local_agent_response_list_dir']['feedback1'] != '':
            data_local['local_agent_response_list_dir']['feedback1'] += '''
                      This is the feedback from local agents.
                      If you find some errors in your previous plan, try to modify it.
                      Otherwise, output the same plan as before.
                      The output should have the same json format {Agent[0.5, 0.5]:move(box_blue, square[0.5, 1.5]), Agent[1.5, 0.5]:move...}, as above.
                      Do not explain, just directly output json directory.
                      Your response:
                      '''
            messages = message_construct_func([data_dict['user_prompt_list'][-1], data_local['local_agent_response_list_dir']['feedback1']],
                                              [response],
                                              '_w_all_dialogue_history')

            # messages = ... # This message should be constructed for teh judge, include both central and local response, agree on global plan
            
            response_central_again, token_num_count = LLaMA_response(messages, model_name)
            
            #-----------------------------------------SYNTACTIC CHECK AGAIN-----------------------------------------#
            data_dict['token_num_count_list'].append(token_num_count)
            match = re.search(r'{.*}', response_central_again, re.DOTALL)
            if match:
              response = match.group()
              response, token_num_count_list_add = with_action_syntactic_check_func(data_dict['pg_dict'],
                                                                                    response_central_again, 
                                                                                    [data_dict['user_prompt_list'][-1], data_local['local_agent_response_list_dir']['feedback1']], 
                                                                                    [response],
                                                                                    model_name,
                                                                                    '_w_all_dialogue_history'
                                                                                    )
              data_dict['token_num_count_list'] = data_dict['token_num_count_list'] + token_num_count_list_add
              print(f'response: {response}')

            # print(messages[2])
            # print(messages[3])
            print(f'MODIFIED:\n {response}')
          else:
            print(f'PLAN:\n {response}')
            pass
          data_dict['dialogue_history_list'].append(dialogue_history)
        
        # print(agent_response_list)
        
  #-----------------------------------------TASK SUCCESS CHECK-----------------------------------------#
  if index_query_times < query_time_limit - 1:
    success_failure = 'success'
  else:
    success_failure = 'failure over query time limit'
  print(success_failure)



#-----------------------------------------RUNNING EXPERIMENT-----------------------------------------#
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
Code_dir_path = os.path.join(os.getcwd())
os.makedirs(Code_dir_path, exist_ok=True)
saving_path = Code_dir_path + '/multi-agent-env'

pg_row_num = 2
pg_column_num = 2
iteration_num = 2
query_time_limit = 10
model_name = "qwen2.5:14b-instruct-q3_K_L"
print(f'-------------------Model name: {model_name}-------------------')

run_exp(saving_path, pg_row_num, pg_column_num, iteration_num, 
        query_time_limit, dialogue_history_method='_w_only_state_action_history'
        )