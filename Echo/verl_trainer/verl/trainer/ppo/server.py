import json
import os
import threading
import time
import base64
import pickle
from SimpleWebSocketServer import SimpleWebSocketServer, WebSocket
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
import queue
import socket

for var in [
    "http_proxy", "https_proxy", "all_proxy", "socks_proxy",
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"
]:
    os.environ.pop(var, None)

ROLLOUTS_DIR = "/opt/projects/verl/data/server_rollouts"
MODEL_DIR = "/opt/projects/verl/data/server_models"
os.makedirs(ROLLOUTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

clients = {}
rollouts_dict = {}  # index -> rollouts_info
rollouts_index = 0
pending_requests = {}  # index -> list of clients waiting for this index

models = {}  # step -> model_info
current_model_step = None  
pending_model_requests = []

SERVER_HOST = "0.0.0.0"  
HTTP_PORT = 8000
SERVER_PORT = 8765
SERVER_IP = "10.0.2.111"  
ROLLOUTS_DIR = "/opt/projects/verl/data/server_rollouts"

def get_local_ip():
    return SERVER_IP

class HTTPHandler(SimpleHTTPRequestHandler):
    """HTTP handler for serving model files."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=MODEL_DIR, **kwargs)
    

def start_http_server():
    """Start HTTP server for serving model files."""
    httpd = HTTPServer(('0.0.0.0', HTTP_PORT), HTTPHandler)
    print(f"[Server] HTTP server running at http://{SERVER_IP}:{HTTP_PORT}")
    print(f"[Server] Serving directory: {MODEL_DIR}")
    httpd.serve_forever()

class WSHandler(WebSocket):
    def handleConnected(self):
        client_address = self.address[0]
        print(f"[Server] New connection from {client_address}")
    
    def handleMessage(self):
        global rollouts_index
        global current_model_step
        
        if self not in clients.values():
            identity = self.data
            clients[identity] = self
            client_address = self.address[0]
            print(f"[Server] {identity} connected from {client_address}")
            return
            
        try:
            data = json.loads(self.data)
            client_address = self.address[0]
            
            if data["type"] == "upload_rollouts":
                filename = data["filename"]
                file_data = data["file_data"]
                batch_idx = data["batch_idx"]
                dataset_name = data["dataset_name"]

                decoded_data = base64.b64decode(file_data)

                dataset_dir = Path(ROLLOUTS_DIR) / dataset_name
                dataset_dir.mkdir(parents=True, exist_ok=True)
                
                file_path = dataset_dir / filename
                with open(file_path, "wb") as f:
                    f.write(decoded_data)
                
                current_index = rollouts_index
                rollouts_info = {
                    "filename": filename,
                    "file_path": str(file_path),
                    "batch_idx": batch_idx,
                    "dataset_name": dataset_name,
                    "index": current_index,
                    "upload_time": time.time(),
                    "client_ip": client_address
                }

                rollouts_dict[current_index] = rollouts_info
                print(f"[Server] Received rollouts {filename} from inference node ({client_address})")
                print(f"[Server] Saved to {file_path} with index: {current_index}")
                
                self.sendMessage(json.dumps({
                    "type": "upload_success",
                    "filename": filename,
                    "message": "Rollouts uploaded successfully",
                    "server_ip": SERVER_IP,
                    "index": current_index
                }))

                if current_index in pending_requests:
                    waiting_clients = pending_requests[current_index]
                    for waiting_client in waiting_clients:
                        try:
                            self._send_rollouts_by_index(waiting_client, current_index)
                        except Exception as e:
                            print(f"[Server] Failed to send rollouts to waiting client: {e}")
                    del pending_requests[current_index]
                
                rollouts_index += 1
        
            elif data["type"] == "request_rollouts":
                requested_index = data["index"]
                timeout = data.get("timeout", 60)
                
                print(f"[Server] Trainer ({client_address}) requesting rollouts index: {requested_index}")
                
                if requested_index in rollouts_dict:
                    self._send_rollouts_by_index(self, requested_index)
                else:
                    if requested_index not in pending_requests:
                        pending_requests[requested_index] = []
                    pending_requests[requested_index].append(self)
                    
                    print(f"[Server] Rollouts index {requested_index} not ready, adding to pending list")
                    
                    def timeout_handler():
                        time.sleep(timeout)
                        if requested_index in pending_requests and self in pending_requests[requested_index]:
                            pending_requests[requested_index].remove(self)
                            if not pending_requests[requested_index]:
                                del pending_requests[requested_index]
                            try:
                                self.sendMessage(json.dumps({
                                    "type": "rollouts_timeout",
                                    "message": f"Timeout waiting for rollouts index {requested_index}",
                                    "server_ip": SERVER_IP,
                                    "requested_index": requested_index
                                }))
                            except:
                                pass
                    
                    timeout_thread = threading.Thread(target=timeout_handler, daemon=True)
                    timeout_thread.start()

            elif data["type"] == "model_ready_notification":
                step = data["step"]
                filename = data["filename"]
                
                model_url = f"http://{SERVER_IP}:{HTTP_PORT}/{filename}"
                
                print(f"[Server] Received model ready notification from trainer ({client_address})")

                model_file_path = os.path.join(MODEL_DIR, filename)
                if not os.path.exists(model_file_path):
                    print(f"[Server] Warning: Model file not found at {model_file_path}")
                    self.sendMessage(json.dumps({
                        "type": "model_notification_error",
                        "step": step,
                        "message": f"Model file not found: {filename}",
                        "server_ip": SERVER_IP
                    }))
                    return

                if current_model_step is not None and current_model_step in models:
                    old_model_info = models[current_model_step]
                    old_model_file = old_model_info.get("file_path")
                    if old_model_file and os.path.exists(old_model_file):
                        try:
                            os.unlink(old_model_file)
                            print(f"[Server] Deleted old model file: {old_model_file}")
                        except Exception as e:
                            print(f"[Server] Failed to delete old model file: {e}")
                    del models[current_model_step]
                
                model_info = {
                    "step": step,
                    "filename": filename,
                    "file_path": model_file_path,
                    "url": model_url,
                    "upload_time": time.time(),
                    "trainer_ip": client_address
                }
                
                models[step] = model_info
                current_model_step = step
                
                print(f"[Server] Model available at: {model_url}")
                
                self.sendMessage(json.dumps({
                    "type": "model_notification_success",
                    "step": step,
                    "filename": filename,
                    "url": model_url,
                    "message": "Model notification received successfully",
                    "server_ip": SERVER_IP
                }))
                
                if pending_model_requests:
                    waiting_clients = pending_model_requests.copy()
                    pending_model_requests.clear()
                    
                    for waiting_client in waiting_clients:
                        try:
                            self._send_model_url_to_client(waiting_client)
                        except Exception as e:
                            print(f"[Server] Failed to send model URL to waiting client: {e}")


            elif data["type"] == "upload_model":
                step = data["step"]
                model_data = data["model_data"]
                format_type = data.get("format", "safetensors")

                decoded_data = base64.b64decode(model_data)

                model_dir = MODEL_DIR
                model_filename = f"model_step_{step}.{format_type}"
                model_path = model_dir / model_filename

                with open(model_path, "wb") as f:
                    f.write(decoded_data)

                model_info = {
                    "step": step,
                    "filename": filename,
                    "file_path": str(model_path),
                    "upload_time": time.time(),
                    "trainer_ip": client_address
                }
                
                models[step] = model_info
                current_model_step = step

                print(f"[Server] Received model from trainer ({client_address})")
                
                self.sendMessage(json.dumps({
                    "type": "model_upload_success",
                    "step": step,
                    "filename": filename,
                    "message": "Model uploaded successfully",
                    "server_ip": SERVER_IP
                }))

                if pending_model_requests:
                    waiting_clients = pending_model_requests.copy()
                    pending_model_requests.clear()
                    
                    for waiting_client in waiting_clients:
                        try:
                            self._send_model_url_to_client(waiting_client)
                        except Exception as e:
                            print(f"[Server] Failed to send model to waiting client: {e}")

            elif data["type"] == "request_latest_model":
                timeout = data.get("timeout", 300)  
                print(f"[Server] Inference node ({client_address}) requesting latest model")

                pending_model_requests.append(self)
                print(f"[Server] No model available, adding inference node to pending list")
                print(f"[Server] Total pending model requests: {len(pending_model_requests)}")
                
                def model_timeout_handler():
                    time.sleep(timeout)
                    if self in pending_model_requests:
                        pending_model_requests.remove(self)
                        try:
                            self.sendMessage(json.dumps({
                                "type": "model_timeout",
                                "message": f"Timeout waiting for model update",
                                "server_ip": SERVER_IP,
                                "timeout_duration": timeout
                            }))
                            print(f"[Server] Model request timeout for inference node ({client_address})")
                        except:
                            pass
                
                timeout_thread = threading.Thread(target=model_timeout_handler, daemon=True)
                timeout_thread.start()

            elif data["type"] == "model_loaded_confirmation":
                step = data["step"]
                print(f"[Server] Inference node ({client_address}) confirmed model step {step} loaded")
                
                if step in models:
                    model_info = models[step]
                    model_path = Path(model_info["file_path"])
                    
                    if model_path.exists():
                        try:
                            model_path.unlink()
                            print(f"[Server] Deleted model file: {model_path}")
                            
                            models[step]["file_deleted"] = True
                            models[step]["delete_time"] = time.time()
                            
                            self.sendMessage(json.dumps({
                                "type": "model_file_deleted",
                                "step": step,
                                "message": "Model file deleted successfully",
                                "server_ip": SERVER_IP
                            }))
                            
                        except Exception as e:
                            print(f"[Server] Failed to delete model file: {e}")
                            self.sendMessage(json.dumps({
                                "type": "model_delete_error",
                                "step": step,
                                "message": f"Failed to delete model file: {str(e)}",
                                "server_ip": SERVER_IP
                            }))
                    else:
                        print(f"[Server] Model file {model_path} already deleted or not found")
                else:
                    print(f"[Server] Model step {step} not found in records")

            elif data["type"] == "server_status":
                status = {
                    "type": "server_status_response",
                    "available_rollouts": list(rollouts_dict.keys()),
                    "current_index": rollouts_index,
                    "connected_clients": list(clients.keys()),
                    "rollouts_dir": ROLLOUTS_DIR,
                    "server_ip": SERVER_IP,
                    "server_host": SERVER_HOST,
                    "server_port": SERVER_PORT,
                    "pending_requests": {str(k): len(v) for k, v in pending_requests.items()},
                    "client_addresses": {identity: client.address[0] for identity, client in clients.items()}
                }
                self.sendMessage(json.dumps(status))
                
            elif data["type"] == "ping":
                # 心跳检测
                self.sendMessage(json.dumps({
                    "type": "pong",
                    "server_ip": SERVER_IP,
                    "timestamp": time.time(),
                    "client_ip": client_address
                }))
                
            elif data["type"] == "list_rollouts":
                rollouts_files = []
                for root, dirs, files in os.walk(ROLLOUTS_DIR):
                    for file in files:
                        if file.endswith('.pt'):
                            file_path = os.path.join(root, file)
                            try:
                                stat = os.stat(file_path)
                                rollouts_files.append({
                                    "filename": file,
                                    "path": file_path,
                                    "size": stat.st_size,
                                    "created_time": stat.st_mtime
                                })
                            except Exception as e:
                                print(f"[Server] Error reading file {file_path}: {e}")
                
                response = {
                    "type": "rollouts_list",
                    "files": rollouts_files,
                    "total_count": len(rollouts_files),
                    "server_ip": SERVER_IP
                }
                self.sendMessage(json.dumps(response))

        except Exception as e:
            print(f"[Server] Error handling message from {client_address}: {e}")
            import traceback
            print(traceback.format_exc())
            self.sendMessage(json.dumps({
                "type": "error",
                "message": str(e),
                "server_ip": SERVER_IP
            }))

    def _send_rollouts_by_index(self, client_socket, index):
        """Send rollouts data to client by index."""
        try:
            rollouts_info = rollouts_dict[index]

            if not os.path.exists(rollouts_info["file_path"]):
                raise FileNotFoundError(f"File {rollouts_info['file_path']} not found")
            
            with open(rollouts_info["file_path"], "rb") as f:
                file_data = f.read()
            
            encoded_data = base64.b64encode(file_data).decode('utf-8')
            
            response = {
                "type": "rollouts_data",
                "filename": rollouts_info["filename"],
                "file_data": encoded_data,
                "batch_idx": rollouts_info["batch_idx"],
                "dataset_name": rollouts_info["dataset_name"],
                "index": rollouts_info["index"],
                "server_ip": SERVER_IP,
                "file_size": len(file_data)
            }
            
            client_socket.sendMessage(json.dumps(response))
            print(f"[Server] Sent rollouts index {index} ({rollouts_info['filename']}) to trainer")
            
        except Exception as e:
            print(f"[Server] Error sending rollouts index {index}: {e}")
            try:
                client_socket.sendMessage(json.dumps({
                    "type": "error",
                    "message": f"Failed to send rollouts index {index}: {str(e)}",
                    "server_ip": SERVER_IP,
                    "requested_index": index
                }))
            except:
                pass

    def _send_model_url_to_client(self, client_socket):
        """Send model url to client."""
        try:
            if current_model_step is None or current_model_step not in models:
                raise ValueError("No model available")
            
            model_info = models[current_model_step]
            
            if not os.path.exists(model_info["file_path"]):
                raise FileNotFoundError(f"Model file {model_info['file_path']} not found")
            
            response = {
                "type": "model_url_data",
                "step": current_model_step,
                "filename": model_info["filename"],
                "url": model_info["url"],
                "server_ip": SERVER_IP
            }
            
            client_socket.sendMessage(json.dumps(response))
            print(f"[Server] Sent model URL to client: {model_info['url']}")
            
        except Exception as e:
            print(f"[Server] Error sending model URL: {e}")
            try:
                client_socket.sendMessage(json.dumps({
                    "type": "model_send_error",
                    "message": f"Failed to send model URL: {str(e)}",
                    "server_ip": SERVER_IP
                }))
            except:
                pass

    def handleClose(self):
        client_address = self.address[0]
        disconnected_identity = None
        for identity, client in list(clients.items()):
            if client == self:
                disconnected_identity = identity
                break
                
        if disconnected_identity:
            print(f"[Server] {disconnected_identity} from {client_address} disconnected.")
            del clients[disconnected_identity]

        for index, waiting_clients in list(pending_requests.items()):
            if self in waiting_clients:
                waiting_clients.remove(self)
                if not waiting_clients:
                    del pending_requests[index]

        if self in pending_model_requests:
            pending_model_requests.remove(self)
            print(f"[Server] Removed disconnected client from pending model requests")

if __name__ == "__main__":
    print(f"[Server] Starting VERL model sync server...")
    print(f"[Server] Server IP: {SERVER_IP}")
    print(f"[Server] HTTP server will serve from: {MODEL_DIR}")
    print(f"[Server] WebSocket server on: {SERVER_HOST}:{SERVER_PORT}")
    
    http_thread = threading.Thread(target=start_http_server, daemon=True)
    http_thread.start()
    time.sleep(2)

    try:
        import requests
        test_url = f"http://{SERVER_IP}:{HTTP_PORT}/"
        response = requests.get(test_url, timeout=5)
        print(f"[Server] HTTP server test successful: {response.status_code}")
    except Exception as e:
        print(f"[Server] HTTP server test failed: {e}")
        print(f"[Server] HTTP server may not be accessible")
    
    server = SimpleWebSocketServer(SERVER_HOST, SERVER_PORT, WSHandler)
    print(f"[Server] WebSocket server running on ws://{SERVER_IP}:{SERVER_PORT}")
    print(f"[Server] HTTP file server running on http://{SERVER_IP}:{HTTP_PORT}")
    
    try:
        server.serveforever()
    except KeyboardInterrupt:
        print(f"\n[Server] Server shutdown")