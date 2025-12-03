#!/usr/bin/env python3
import requests

# 两台机器配置
SERVICE_IP = "106.14.183.16"           # work1: service_api (8099端口)
MODEL_ECHO_IP = "14.103.221.215"       # 模型机: vLLM(8080端口) + Echo(8081端口)

# 定义全部86个任务
ALL_TASKS = [
    # social_network (25个)
    "k8s_target_port-misconfig-detection-1",
    "k8s_target_port-misconfig-detection-2",
    "k8s_target_port-misconfig-detection-3",
    "auth_miss_mongodb-detection-1",
    "scale_pod_zero_social_net-detection-1",
    "assign_to_non_existent_node_social_net-detection-1",
    "noop_detection_social_network-1",
    "k8s_target_port-misconfig-localization-1",
    "k8s_target_port-misconfig-localization-2",
    "k8s_target_port-misconfig-localization-3",
    "auth_miss_mongodb-localization-1",
    "scale_pod_zero_social_net-localization-1",
    "assign_to_non_existent_node_social_net-localization-1",
    "k8s_target_port-misconfig-analysis-1",
    "k8s_target_port-misconfig-analysis-2",
    "k8s_target_port-misconfig-analysis-3",
    "auth_miss_mongodb-analysis-1",
    "scale_pod_zero_social_net-analysis-1",
    "assign_to_non_existent_node_social_net-analysis-1",
    "k8s_target_port-misconfig-mitigation-1",
    "k8s_target_port-misconfig-mitigation-2",
    "k8s_target_port-misconfig-mitigation-3",
    "auth_miss_mongodb-mitigation-1",
    "scale_pod_zero_social_net-mitigation-1",
    "assign_to_non_existent_node_social_net-mitigation-1",
    
    # hotel_reservation (38个)
    "revoke_auth_mongodb-detection-1",
    "revoke_auth_mongodb-detection-2",
    "user_unregistered_mongodb-detection-1",
    "user_unregistered_mongodb-detection-2",
    "misconfig_app_hotel_res-detection-1",
    "container_kill-detection",
    "pod_failure_hotel_res-detection-1",
    "pod_kill_hotel_res-detection-1",
    "network_loss_hotel_res-detection-1",
    "network_delay_hotel_res-detection-1",
    "redeploy_without_PV-detection-1",
    "wrong_bin_usage-detection-1",
    "noop_detection_hotel_reservation-1",
    "revoke_auth_mongodb-localization-1",
    "revoke_auth_mongodb-localization-2",
    "user_unregistered_mongodb-localization-1",
    "user_unregistered_mongodb-localization-2",
    "misconfig_app_hotel_res-localization-1",
    "container_kill-localization",
    "pod_failure_hotel_res-localization-1",
    "pod_kill_hotel_res-localization-1",
    "network_loss_hotel_res-localization-1",
    "network_delay_hotel_res-localization-1",
    "wrong_bin_usage-localization-1",
    "revoke_auth_mongodb-analysis-1",
    "revoke_auth_mongodb-analysis-2",
    "user_unregistered_mongodb-analysis-1",
    "user_unregistered_mongodb-analysis-2",
    "misconfig_app_hotel_res-analysis-1",
    "redeploy_without_PV-analysis-1",
    "wrong_bin_usage-analysis-1",
    "revoke_auth_mongodb-mitigation-1",
    "revoke_auth_mongodb-mitigation-2",
    "user_unregistered_mongodb-mitigation-1",
    "user_unregistered_mongodb-mitigation-2",
    "misconfig_app_hotel_res-mitigation-1",
    "redeploy_without_PV-mitigation-1",
    "wrong_bin_usage-mitigation-1",
    
    # astronomy_shop (23个)
    "astronomy_shop_ad_service_failure-detection-1",
    "astronomy_shop_ad_service_high_cpu-detection-1",
    "astronomy_shop_ad_service_manual_gc-detection-1",
    "astronomy_shop_cart_service_failure-detection-1",
    "astronomy_shop_image_slow_load-detection-1",
    "astronomy_shop_kafka_queue_problems-detection-1",
    "astronomy_shop_loadgenerator_flood_homepage-detection-1",
    "astronomy_shop_payment_service_failure-detection-1",
    "astronomy_shop_payment_service_unreachable-detection-1",
    "astronomy_shop_product_catalog_service_failure-detection-1",
    "astronomy_shop_recommendation_service_cache_failure-detection-1",
    "noop_detection_astronomy_shop-1",
    "astronomy_shop_ad_service_failure-localization-1",
    "astronomy_shop_ad_service_high_cpu-localization-1",
    "astronomy_shop_ad_service_manual_gc-localization-1",
    "astronomy_shop_cart_service_failure-localization-1",
    "astronomy_shop_image_slow_load-localization-1",
    "astronomy_shop_kafka_queue_problems-localization-1",
    "astronomy_shop_loadgenerator_flood_homepage-localization-1",
    "astronomy_shop_payment_service_failure-localization-1",
    "astronomy_shop_payment_service_unreachable-localization-1",
    "astronomy_shop_product_catalog_service_failure-localization-1",
    "astronomy_shop_recommendation_service_cache_failure-localization-1",
]

# 🔧 修改点: 选择要运行的任务（可以调整范围）

# TASKS_TO_RUN = ALL_TASKS[0:1]   
# TASKS_TO_RUN = ALL_TASKS[25:26] 
# TASKS_TO_RUN = ALL_TASKS[63:64]   
   
TASKS_TO_RUN = ALL_TASKS[63:68]      
# TASKS_TO_RUN = ALL_TASKS          # 如果要跑全部86个任务，取消这行注释

payload = {
    "problems": [
        {"problem_id": task_id, "runs": 1, "max_steps": 12}
        for task_id in TASKS_TO_RUN
    ],
    "concurrency": 5,  # 并发数（可根据任务数量调整）
    
    "chat": {
        "model": "/data1/xj/pred_model/Qwen2.5-Coder-14B-Instruct",
        "base_url": f"http://{MODEL_ECHO_IP}:18200/v1",  
        "temperature": 0.2,
        "top_p": 1.0,
        "max_tokens": 512
    },
    "echo": {
        "url": f"http://{MODEL_ECHO_IP}:18209"
    }
}

print(f"准备发送 {len(TASKS_TO_RUN)} 个任务到 {SERVICE_IP}:")
for i, task_id in enumerate(TASKS_TO_RUN, 1):
    print(f"  {i}. {task_id}")
print()

resp = requests.post(
    f"http://{SERVICE_IP}:8099/echo/jobs",
    json=payload,
    timeout=60,
)
resp.raise_for_status()
print("任务提交成功！")
print(resp.json())


