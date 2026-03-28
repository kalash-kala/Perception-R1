import pandas as pd
import numpy as np
import copy

def update_acceptable_answers():
    original_df = pd.read_parquet('/home/sriramg/kalashabhayk/visual-question-answering/vqa_stratified_100.parquet')
    
    paths = [
        '/home/sriramg/kalashabhayk/visual-question-answering/processed_for_verl/train_perturbed_vqa.parquet',
        '/home/sriramg/kalashabhayk/visual-question-answering/processed_for_verl/val_perturbed_vqa.parquet'
    ]

    for path in paths:
        print(f"Processing {path}...")
        df = pd.read_parquet(path)
        
        new_reward_models = []
        for i, row in df.iterrows():
            extra_info = row['extra_info']
            source_id = int(extra_info['source_id'])
            
            # Fetch from original table by row index
            orig_answers = original_df['answers'].iloc[source_id]
            
            # If it's a numpy array, convert to list.
            if isinstance(orig_answers, np.ndarray):
                orig_answers = orig_answers.tolist()
                
            # Extract just the answer strings
            extracted_answers = [ans['answer'] for ans in orig_answers if ans['answer_confidence'] in ['yes']]
            
            # Update the reward model dictonary
            rm = copy.deepcopy(row['reward_model'])
            # Keeping the key as acceptable_answers based on the structure you showed
            if rm.get('answerability') == 'UNANSWERABLE':
                rm['acceptable_answers'] = ["I don't know"]
            else:
                rm['acceptable_answers'] = extracted_answers
            
            new_reward_models.append(rm)
            
        df['reward_model'] = new_reward_models
        df.to_parquet(path)
        print(f"Successfully updated and saved {path}")

if __name__ == "__main__":
    update_acceptable_answers()
    print("All done!")
