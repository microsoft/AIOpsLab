# Problem Registry Identifiers

The orchestrator exposes the following problem identifiers through
`ProblemRegistry` (`aiopslab/orchestrator/problems/registry.py`). They can be
used when creating jobs via the Echo callback server or when resetting
environments through `service_api`.

Total entries: **178**. The registry automatically adds
canonical identifiers (without numeric suffixes) alongside the numbered
variants, so both forms appear below in insertion order.

- `k8s_target_port-misconfig-detection-1`
- `k8s_target_port-misconfig-localization-1`
- `k8s_target_port-misconfig-analysis-1`
- `k8s_target_port-misconfig-mitigation-1`
- `k8s_target_port-misconfig-detection-2`
- `k8s_target_port-misconfig-localization-2`
- `k8s_target_port-misconfig-analysis-2`
- `k8s_target_port-misconfig-mitigation-2`
- `k8s_target_port-misconfig-detection-3`
- `k8s_target_port-misconfig-localization-3`
- `k8s_target_port-misconfig-analysis-3`
- `k8s_target_port-misconfig-mitigation-3`
- `auth_miss_mongodb-detection-1`
- `auth_miss_mongodb-localization-1`
- `auth_miss_mongodb-analysis-1`
- `auth_miss_mongodb-mitigation-1`
- `revoke_auth_mongodb-detection-1`
- `revoke_auth_mongodb-localization-1`
- `revoke_auth_mongodb-analysis-1`
- `revoke_auth_mongodb-mitigation-1`
- `revoke_auth_mongodb-detection-2`
- `revoke_auth_mongodb-localization-2`
- `revoke_auth_mongodb-analysis-2`
- `revoke_auth_mongodb-mitigation-2`
- `user_unregistered_mongodb-detection-1`
- `user_unregistered_mongodb-localization-1`
- `user_unregistered_mongodb-analysis-1`
- `user_unregistered_mongodb-mitigation-1`
- `user_unregistered_mongodb-detection-2`
- `user_unregistered_mongodb-localization-2`
- `user_unregistered_mongodb-analysis-2`
- `user_unregistered_mongodb-mitigation-2`
- `misconfig_app_hotel_res-detection-1`
- `misconfig_app_hotel_res-localization-1`
- `misconfig_app_hotel_res-analysis-1`
- `misconfig_app_hotel_res-mitigation-1`
- `scale_pod_zero_social_net-detection-1`
- `scale_pod_zero_social_net-localization-1`
- `scale_pod_zero_social_net-analysis-1`
- `scale_pod_zero_social_net-mitigation-1`
- `assign_to_non_existent_node_social_net-detection-1`
- `assign_to_non_existent_node_social_net-localization-1`
- `assign_to_non_existent_node_social_net-analysis-1`
- `assign_to_non_existent_node_social_net-mitigation-1`
- `container_kill-detection`
- `container_kill-detection-1`
- `container_kill-localization`
- `container_kill-localization-1`
- `container_kill-analysis`
- `container_kill-analysis-1`
- `container_kill-mitigation`
- `container_kill-mitigation-1`
- `pod_failure_hotel_res-detection-1`
- `pod_failure_hotel_res-localization-1`
- `pod_failure_hotel_res-analysis-1`
- `pod_failure_hotel_res-mitigation-1`
- `pod_kill_hotel_res-detection-1`
- `pod_kill_hotel_res-localization-1`
- `pod_kill_hotel_res-analysis-1`
- `pod_kill_hotel_res-mitigation-1`
- `network_loss_hotel_res-detection-1`
- `network_loss_hotel_res-localization-1`
- `network_loss_hotel_res-analysis-1`
- `network_loss_hotel_res-mitigation-1`
- `network_delay_hotel_res-detection-1`
- `network_delay_hotel_res-localization-1`
- `network_delay_hotel_res-analysis-1`
- `network_delay_hotel_res-mitigation-1`
- `noop_detection_hotel_reservation-1`
- `noop_detection_social_network-1`
- `noop_detection_astronomy_shop-1`
- `astronomy_shop_ad_service_failure-detection-1`
- `astronomy_shop_ad_service_failure-localization-1`
- `astronomy_shop_ad_service_high_cpu-detection-1`
- `astronomy_shop_ad_service_high_cpu-localization-1`
- `astronomy_shop_ad_service_manual_gc-detection-1`
- `astronomy_shop_ad_service_manual_gc-localization-1`
- `astronomy_shop_cart_service_failure-detection-1`
- `astronomy_shop_cart_service_failure-localization-1`
- `astronomy_shop_image_slow_load-detection-1`
- `astronomy_shop_image_slow_load-localization-1`
- `astronomy_shop_kafka_queue_problems-detection-1`
- `astronomy_shop_kafka_queue_problems-localization-1`
- `astronomy_shop_loadgenerator_flood_homepage-detection-1`
- `astronomy_shop_loadgenerator_flood_homepage-localization-1`
- `astronomy_shop_payment_service_failure-detection-1`
- `astronomy_shop_payment_service_failure-localization-1`
- `astronomy_shop_payment_service_unreachable-detection-1`
- `astronomy_shop_payment_service_unreachable-localization-1`
- `astronomy_shop_product_catalog_service_failure-detection-1`
- `astronomy_shop_product_catalog_service_failure-localization-1`
- `astronomy_shop_recommendation_service_cache_failure-detection-1`
- `astronomy_shop_recommendation_service_cache_failure-localization-1`
- `redeploy_without_PV-detection-1`
- `redeploy_without_PV-analysis-1`
- `redeploy_without_PV-mitigation-1`
- `wrong_bin_usage-detection-1`
- `wrong_bin_usage-localization-1`
- `wrong_bin_usage-analysis-1`
- `wrong_bin_usage-mitigation-1`
- `flower_node_stop-detection`
- `flower_model_misconfig-detection`
- `k8s_target_port-misconfig-detection`
- `k8s_target_port-misconfig-localization`
- `k8s_target_port-misconfig-analysis`
- `k8s_target_port-misconfig-mitigation`
- `auth_miss_mongodb-detection`
- `auth_miss_mongodb-localization`
- `auth_miss_mongodb-analysis`
- `auth_miss_mongodb-mitigation`
- `revoke_auth_mongodb-detection`
- `revoke_auth_mongodb-localization`
- `revoke_auth_mongodb-analysis`
- `revoke_auth_mongodb-mitigation`
- `user_unregistered_mongodb-detection`
- `user_unregistered_mongodb-localization`
- `user_unregistered_mongodb-analysis`
- `user_unregistered_mongodb-mitigation`
- `misconfig_app_hotel_res-detection`
- `misconfig_app_hotel_res-localization`
- `misconfig_app_hotel_res-analysis`
- `misconfig_app_hotel_res-mitigation`
- `scale_pod_zero_social_net-detection`
- `scale_pod_zero_social_net-localization`
- `scale_pod_zero_social_net-analysis`
- `scale_pod_zero_social_net-mitigation`
- `assign_to_non_existent_node_social_net-detection`
- `assign_to_non_existent_node_social_net-localization`
- `assign_to_non_existent_node_social_net-analysis`
- `assign_to_non_existent_node_social_net-mitigation`
- `pod_failure_hotel_res-detection`
- `pod_failure_hotel_res-localization`
- `pod_failure_hotel_res-analysis`
- `pod_failure_hotel_res-mitigation`
- `pod_kill_hotel_res-detection`
- `pod_kill_hotel_res-localization`
- `pod_kill_hotel_res-analysis`
- `pod_kill_hotel_res-mitigation`
- `network_loss_hotel_res-detection`
- `network_loss_hotel_res-localization`
- `network_loss_hotel_res-analysis`
- `network_loss_hotel_res-mitigation`
- `network_delay_hotel_res-detection`
- `network_delay_hotel_res-localization`
- `network_delay_hotel_res-analysis`
- `network_delay_hotel_res-mitigation`
- `noop_detection_hotel_reservation`
- `noop_detection_social_network`
- `noop_detection_astronomy_shop`
- `astronomy_shop_ad_service_failure-detection`
- `astronomy_shop_ad_service_failure-localization`
- `astronomy_shop_ad_service_high_cpu-detection`
- `astronomy_shop_ad_service_high_cpu-localization`
- `astronomy_shop_ad_service_manual_gc-detection`
- `astronomy_shop_ad_service_manual_gc-localization`
- `astronomy_shop_cart_service_failure-detection`
- `astronomy_shop_cart_service_failure-localization`
- `astronomy_shop_image_slow_load-detection`
- `astronomy_shop_image_slow_load-localization`
- `astronomy_shop_kafka_queue_problems-detection`
- `astronomy_shop_kafka_queue_problems-localization`
- `astronomy_shop_loadgenerator_flood_homepage-detection`
- `astronomy_shop_loadgenerator_flood_homepage-localization`
- `astronomy_shop_payment_service_failure-detection`
- `astronomy_shop_payment_service_failure-localization`
- `astronomy_shop_payment_service_unreachable-detection`
- `astronomy_shop_payment_service_unreachable-localization`
- `astronomy_shop_product_catalog_service_failure-detection`
- `astronomy_shop_product_catalog_service_failure-localization`
- `astronomy_shop_recommendation_service_cache_failure-detection`
- `astronomy_shop_recommendation_service_cache_failure-localization`
- `redeploy_without_PV-detection`
- `redeploy_without_PV-analysis`
- `redeploy_without_PV-mitigation`
- `wrong_bin_usage-detection`
- `wrong_bin_usage-localization`
- `wrong_bin_usage-analysis`
- `wrong_bin_usage-mitigation`

## Regenerating the List

If new problems are registered, regenerate the bullets with:

```bash
python - <<'PY'
import ast
from pathlib import Path
source = Path('aiopslab/orchestrator/problems/registry.py').read_text()
module = ast.parse(source)
keys = []
class Visitor(ast.NodeVisitor):
    def visit_AnnAssign(self, node):
        target = node.target
        if isinstance(target, ast.Attribute) and target.attr == '_static_registry':
            value = node.value
            if isinstance(value, ast.Dict):
                for key in value.keys:
                    if isinstance(key, ast.Constant) and isinstance(key.value, str):
                        keys.append(key.value)
        self.generic_visit(node)
Visitor().visit(module)
print(f"Total entries: {len(keys)}")
for key in keys:
    print(f"- `{key}`")
PY
```

Update the markdown with the new output.
