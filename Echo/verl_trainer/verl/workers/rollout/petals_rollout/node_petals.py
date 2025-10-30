import json
import requests
import os
import torch
from typing import List
from petals_infer import PetalsInferencer
import websocket  # 使用同步WebSocket客户端库

for var in [
    "http_proxy", "https_proxy", "all_proxy", "socks_proxy",
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"
]:
    os.environ.pop(var, None)

MODEL_DIR = "model"
ROLLOUTS_DIR = "model"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(ROLLOUTS_DIR, exist_ok=True)

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
INITIAL_PEERS = [
    "/ip4/10.0.0.128/tcp/32801/p2p/12D3KooWM1Dz1K2unSSnDzWssiph6nyQ138po1fbeHeTugrrTWkf",
    "/ip4/127.0.0.1/tcp/32801/p2p/12D3KooWM1Dz1K2unSSnDzWssiph6nyQ138po1fbeHeTugrrTWkf",
]

petals_inferencer = PetalsInferencer(model_name=MODEL_NAME, initial_peers=INITIAL_PEERS)

def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids

def main():
    while True:  # 持久连接循环
        try:
            uri = "ws://10.0.2.111:8765"
            print(f"正在连接服务器 {uri}...")
            
            # 创建WebSocket连接
            ws = websocket.create_connection(uri) 
            ws.send("node_petals")

            # 持续接收和处理消息
            while True:
                try:
                    msg = ws.recv()
                    data = json.loads(msg)

                    if data["type"] == "new_model":
                        filename = data["filename"]
                        url = data["url"]

                        os.makedirs(MODEL_DIR, exist_ok=True)
                        filepath = os.path.join(MODEL_DIR, filename)
                        
                        print(f"Petals节点 开始下载模型文件: {url}")
                        response = requests.get(url)
                        with open(filepath, "wb") as f:
                            f.write(response.content)

                        print(f"Petals节点 下载并保存文件: {filename}")

                        if filename.endswith(".safetensors"):
                            print(f"Petals节点 加载safetensors权重: {filename}")
                            petals_inferencer.load_weights(filepath)

                    elif data["type"] == "new_prompts":
                        filename = data["filename"]
                        url = data["url"]
                        
                        print(f"Petals节点 开始下载prompts文件: {url}")
                        try:
                            response = requests.get(url, timeout=30)
                            response.raise_for_status()
                                
                            # 获取文本内容并解析JSON
                            content = response.text
                            print(f"Petals节点 下载内容长度: {len(content)} 字符")
                                
                            prompts_data = json.loads(content)
                            print(f"Petals节点 成功解析JSON，处理 {len(prompts_data)} 个prompts")
                            
                        except requests.exceptions.RequestException as e:
                            print(f"Petals节点 下载错误: {e}")
                            continue
                        except json.JSONDecodeError as e:
                            print(f"Petals节点 JSON解析错误: {e}")
                            print(f"错误位置: line {e.lineno}, column {e.colno}")
                            # 保存有问题的内容到文件以便调试
                            debug_file = f"debug_prompts_{filename}"
                            with open(debug_file, "w", encoding='utf-8') as f:
                                f.write(content)
                            print(f"已保存原始内容到 {debug_file}")
                            continue
                        except Exception as e:
                            print(f"Petals节点 未知错误: {e}")
                            continue

                        results = []
                            
                        for i, prompt_data in enumerate(prompts_data):
                            try:
                                # 判断输入数据类型
                                if isinstance(prompt_data, list):
                                    # 如果是token IDs列表，需要解码
                                    pad_token_id = petals_inferencer.tokenizer.pad_token_id if petals_inferencer.tokenizer.pad_token_id is not None else petals_inferencer.tokenizer.eos_token_id
                                    prompt_tensor = torch.tensor(prompt_data)
                                    processed_ids = _pre_process_inputs(pad_token_id, prompt_tensor)
                                    prompt_text = petals_inferencer.tokenizer.decode(processed_ids, skip_special_tokens=True)
                                else:
                                    # 如果已经是文本，直接使用
                                    prompt_text = str(prompt_data)
                                    
                                print(f"Petals节点 推理输入 {i+1}/{len(prompts_data)}")
                                
                                # 使用预训练模型进行推理
                                output = petals_inferencer.infer(prompt_text)
                                
                                # 确保结果是JSON可序列化的
                                if isinstance(output, torch.Tensor):
                                    output_list = output.detach().cpu().tolist()
                                    results.append(output_list)
                                else:
                                    results.append(output)
                                    
                            except Exception as e:
                                print(f"Petals节点 处理prompt {i+1} 时出错: {e}")
                                # 添加一个默认的响应
                                eos_token_id = petals_inferencer.tokenizer.eos_token_id
                                results.append([eos_token_id] if eos_token_id is not None else [2])  # 2是常见的EOS token
                            
                        # 发送rollouts结果回verl端
                        response_msg = {
                            "type": "rollouts",
                            "filename": filename,
                            "result": results
                        }
                            
                        try:
                            # 测试序列化以确保数据可发送
                            json_data = json.dumps(response_msg)
                            ws.send(json_data)
                            print(f"Petals节点 完成推理，返回 {len(results)} 个结果")
                        except TypeError as e:
                            print(f"序列化错误: {e}，尝试修复...")
                            # 替换不可序列化的内容
                            for i, item in enumerate(results):
                                if not isinstance(item, (str, int, float, bool, list, dict, type(None))):
                                    results[i] = str(item)
                            response_msg["result"] = results
                            ws.send(json.dumps(response_msg))
                
                except websocket.WebSocketConnectionClosedException:
                    print("WebSocket连接已关闭，尝试重新连接...")
                    break
                except json.JSONDecodeError as e:
                    print(f"JSON解析错误: {e}")
                    continue
                except Exception as e:
                    print(f"处理消息时出错: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        except (ConnectionRefusedError, websocket.WebSocketException) as e:
            print(f"连接错误: {e}，30秒后重试...")
            import time
            time.sleep(30) 
        except Exception as e:
            print(f"未预期的错误: {e}")
            import time
            time.sleep(30)

if __name__ == "__main__":
    main()

# import asyncio
# import websockets
# import json
# import requests
# import os
# import torch
# from typing import List
# from petals_infer import PetalsInferencer  

# for var in [
#     "http_proxy", "https_proxy", "all_proxy", "socks_proxy",
#     "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"
# ]:
#     os.environ.pop(var, None)

# MODEL_DIR = "model"
# ROLLOUTS_DIR = "model"

# os.makedirs(MODEL_DIR, exist_ok=True)
# os.makedirs(ROLLOUTS_DIR, exist_ok=True)

# MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
# INITIAL_PEERS = [
#     "/ip4/10.0.0.128/tcp/32801/p2p/12D3KooWM1Dz1K2unSSnDzWssiph6nyQ138po1fbeHeTugrrTWkf",
#     "/ip4/127.0.0.1/tcp/32801/p2p/12D3KooWM1Dz1K2unSSnDzWssiph6nyQ138po1fbeHeTugrrTWkf",
# ]

# petals_inferencer = PetalsInferencer(model_name=MODEL_NAME, initial_peers=INITIAL_PEERS)

# def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
#     non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
#     token_ids = prompt_token_ids[non_pad_index:].tolist()
#     return token_ids

# async def main():
#     uri = "ws://10.0.2.111:8765"
#     async with websockets.connect(uri) as websocket:
#         await websocket.send("node_petals")

#         while True:
#             msg = await websocket.recv()
#             data = json.loads(msg)

#             if data["type"] == "new_model":
#                 filename = data["filename"]
#                 url = data["url"]

#                 os.makedirs(MODEL_DIR, exist_ok=True)
#                 filepath = os.path.join(MODEL_DIR, filename)
                
#                 print(f"Petals节点 开始下载模型文件: {url}")
#                 response = requests.get(url)
#                 with open(filepath, "wb") as f:
#                     f.write(response.content)

#                 print(f"Petals节点 下载并保存文件: {filename}")

#                 if filename.endswith(".safetensors"):
#                     print(f"Petals节点 加载safetensors权重: {filename}")
#                     petals_inferencer.load_weights(filepath)

#             elif data["type"] == "new_prompts":
#                 filename = data["filename"]
#                 url = data["url"]
                
#                 print(f"Petals节点 开始下载prompts文件: {url}")
#                 try:
#                     response = requests.get(url, timeout=30)
#                     response.raise_for_status()
                        
#                      # 获取文本内容并解析JSON
#                     content = response.text
#                     print(f"Petals节点 下载内容长度: {len(content)} 字符")
                        
#                     prompts_data = json.loads(content)
#                     print(f"Petals节点 成功解析JSON，处理 {len(prompts_data)} 个prompts")
                    
#                 except requests.exceptions.RequestException as e:
#                     print(f"Petals节点 下载错误: {e}")
#                     continue
#                 except json.JSONDecodeError as e:
#                     print(f"Petals节点 JSON解析错误: {e}")
#                     print(f"错误位置: line {e.lineno}, column {e.colno}")
#                     # 保存有问题的内容到文件以便调试
#                     debug_file = f"debug_prompts_{filename}"
#                     with open(debug_file, "w", encoding='utf-8') as f:
#                         f.write(content)
#                     print(f"已保存原始内容到 {debug_file}")
#                     continue
#                 except Exception as e:
#                     print(f"Petals节点 未知错误: {e}")
#                     continue

#                 results = []
                    
#                 for i, prompt_data in enumerate(prompts_data):
#                     try:
#                         # 判断输入数据类型
#                         if isinstance(prompt_data, list):
#                             # 如果是token IDs列表，需要解码
#                             pad_token_id = petals_inferencer.tokenizer.pad_token_id if petals_inferencer.tokenizer.pad_token_id is not None else petals_inferencer.tokenizer.eos_token_id
#                             prompt_tensor = torch.tensor(prompt_data)
#                             processed_ids = _pre_process_inputs(pad_token_id, prompt_tensor)
#                             prompt_text = petals_inferencer.tokenizer.decode(processed_ids, skip_special_tokens=True)
#                         else:
#                             # 如果已经是文本，直接使用
#                             prompt_text = str(prompt_data)
                            
#                         # print(f"Petals节点 推理输入 {i+1}/{len(prompts_data)}")
                        
#                         # 使用预训练模型进行推理
#                         output = petals_inferencer.infer(prompt_text)

#                         results.append(output.tolist())
                            
#                     except Exception as e:
#                         print(f"Petals节点 处理prompt {i+1} 时出错: {e}")
#                         # 添加一个默认的响应
#                         eos_token_id = petals_inferencer.tokenizer.eos_token_id
#                         results.append([eos_token_id] if eos_token_id is not None else [2])  # 2是常见的EOS token
                    
#                 # 发送rollouts结果回verl端
#                 response_msg = {
#                     "type": "rollouts",
#                     "filename": filename,
#                     "result": results
#                 }
                    
#                 await websocket.send(json.dumps(response_msg))
#                 print(f"Petals节点 完成推理，返回 {len(results)} 个结果")

# if __name__ == "__main__":
#     asyncio.run(main())