from aiopslab.orchestrator.problems.k8s_target_port_misconfig import *
from aiopslab.orchestrator.problems.auth_miss_mongodb import *
from aiopslab.orchestrator.problems.revoke_auth import *
from aiopslab.orchestrator.problems.storage_user_unregistered import *
from aiopslab.orchestrator.problems.misconfig_app import *
from aiopslab.orchestrator.problems.scale_pod import *
from aiopslab.orchestrator.problems.assign_non_existent_node import *
from aiopslab.orchestrator.problems.container_kill import *
from aiopslab.orchestrator.problems.pod_failure import *
from aiopslab.orchestrator.problems.pod_kill import *
from aiopslab.orchestrator.problems.network_loss import *
from aiopslab.orchestrator.problems.network_delay import *
from aiopslab.orchestrator.problems.no_op import *
from aiopslab.orchestrator.problems.kernel_fault import *
from aiopslab.orchestrator.problems.disk_woreout import *
from aiopslab.orchestrator.problems.ad_service_failure import *
from aiopslab.orchestrator.problems.ad_service_high_cpu import *
from aiopslab.orchestrator.problems.ad_service_manual_gc import *
from aiopslab.orchestrator.problems.cart_service_failure import *
from aiopslab.orchestrator.problems.image_slow_load import *
from aiopslab.orchestrator.problems.kafka_queue_problems import *
from aiopslab.orchestrator.problems.loadgenerator_flood_homepage import *
from aiopslab.orchestrator.problems.payment_service_failure import *
from aiopslab.orchestrator.problems.payment_service_unreachable import *
from aiopslab.orchestrator.problems.product_catalog_failure import *
from aiopslab.orchestrator.problems.recommendation_service_cache_failure import *
from aiopslab.orchestrator.problems.redeploy_without_pv import *
from aiopslab.orchestrator.problems.wrong_bin_usage import *
from aiopslab.orchestrator.problems.operator_misoperation import *
from aiopslab.orchestrator.problems.flower_node_stop import *
from aiopslab.orchestrator.problems.flower_model_misconfig import *


class ProblemRegistry:
    def __init__(self):
        self.PROBLEM_REGISTRY = {
            # K8s target port misconfig
            "k8s_target_port-misconfig-detection-1": lambda: K8STargetPortMisconfigDetection(
                faulty_service="user-service"
            ),
            "k8s_target_port-misconfig-localization-1": lambda: K8STargetPortMisconfigLocalization(
                faulty_service="user-service"
            ),
            "k8s_target_port-misconfig-analysis-1": lambda: K8STargetPortMisconfigAnalysis(
                faulty_service="user-service"
            ),
            "k8s_target_port-misconfig-mitigation-1": lambda: K8STargetPortMisconfigMitigation(
                faulty_service="user-service"
            ),
            "k8s_target_port-misconfig-detection-2": lambda: K8STargetPortMisconfigDetection(
                faulty_service="text-service"
            ),
            "k8s_target_port-misconfig-localization-2": lambda: K8STargetPortMisconfigLocalization(
                faulty_service="text-service"
            ),
            "k8s_target_port-misconfig-analysis-2": lambda: K8STargetPortMisconfigAnalysis(
                faulty_service="text-service"
            ),
            "k8s_target_port-misconfig-mitigation-2": lambda: K8STargetPortMisconfigMitigation(
                faulty_service="text-service"
            ),
            "k8s_target_port-misconfig-detection-3": lambda: K8STargetPortMisconfigDetection(
                faulty_service="post-storage-service"
            ),
            "k8s_target_port-misconfig-localization-3": lambda: K8STargetPortMisconfigLocalization(
                faulty_service="post-storage-service"
            ),
            "k8s_target_port-misconfig-analysis-3": lambda: K8STargetPortMisconfigAnalysis(
                faulty_service="post-storage-service"
            ),
            "k8s_target_port-misconfig-mitigation-3": lambda: K8STargetPortMisconfigMitigation(
                faulty_service="post-storage-service"
            ),
            # MongoDB auth missing
            "auth_miss_mongodb-detection-1": MongoDBAuthMissingDetection,
            "auth_miss_mongodb-localization-1": MongoDBAuthMissingLocalization,
            "auth_miss_mongodb-analysis-1": MongoDBAuthMissingAnalysis,
            "auth_miss_mongodb-mitigation-1": MongoDBAuthMissingMitigation,
            # MongoDB auth revoke
            "revoke_auth_mongodb-detection-1": lambda: MongoDBRevokeAuthDetection(
                faulty_service="mongodb-geo"
            ),
            "revoke_auth_mongodb-localization-1": lambda: MongoDBRevokeAuthLocalization(
                faulty_service="mongodb-geo"
            ),
            "revoke_auth_mongodb-analysis-1": lambda: MongoDBRevokeAuthAnalysis(
                faulty_service="mongodb-geo"
            ),
            "revoke_auth_mongodb-mitigation-1": lambda: MongoDBRevokeAuthMitigation(
                faulty_service="mongodb-geo"
            ),
            "revoke_auth_mongodb-detection-2": lambda: MongoDBRevokeAuthDetection(
                faulty_service="mongodb-rate"
            ),
            "revoke_auth_mongodb-localization-2": lambda: MongoDBRevokeAuthLocalization(
                faulty_service="mongodb-rate"
            ),
            "revoke_auth_mongodb-analysis-2": lambda: MongoDBRevokeAuthAnalysis(
                faulty_service="mongodb-rate"
            ),
            "revoke_auth_mongodb-mitigation-2": lambda: MongoDBRevokeAuthMitigation(
                faulty_service="mongodb-rate"
            ),
            # MongoDB user unregistered
            "user_unregistered_mongodb-detection-1": lambda: MongoDBUserUnregisteredDetection(
                faulty_service="mongodb-geo"
            ),
            "user_unregistered_mongodb-localization-1": lambda: MongoDBUserUnregisteredLocalization(
                faulty_service="mongodb-geo"
            ),
            "user_unregistered_mongodb-analysis-1": lambda: MongoDBUserUnregisteredAnalysis(
                faulty_service="mongodb-geo"
            ),
            "user_unregistered_mongodb-mitigation-1": lambda: MongoDBUserUnregisteredMitigation(
                faulty_service="mongodb-geo"
            ),
            "user_unregistered_mongodb-detection-2": lambda: MongoDBUserUnregisteredDetection(
                faulty_service="mongodb-rate"
            ),
            "user_unregistered_mongodb-localization-2": lambda: MongoDBUserUnregisteredLocalization(
                faulty_service="mongodb-rate"
            ),
            "user_unregistered_mongodb-analysis-2": lambda: MongoDBUserUnregisteredAnalysis(
                faulty_service="mongodb-rate"
            ),
            "user_unregistered_mongodb-mitigation-2": lambda: MongoDBUserUnregisteredMitigation(
                faulty_service="mongodb-rate"
            ),
            # App misconfig
            "misconfig_app_hotel_res-detection-1": MisconfigAppHotelResDetection,
            "misconfig_app_hotel_res-localization-1": MisconfigAppHotelResLocalization,
            "misconfig_app_hotel_res-analysis-1": MisconfigAppHotelResAnalysis,
            "misconfig_app_hotel_res-mitigation-1": MisconfigAppHotelResMitigation,
            # Scale pod to zero deployment
            "scale_pod_zero_social_net-detection-1": ScalePodSocialNetDetection,
            "scale_pod_zero_social_net-localization-1": ScalePodSocialNetLocalization,
            "scale_pod_zero_social_net-analysis-1": ScalePodSocialNetAnalysis,
            "scale_pod_zero_social_net-mitigation-1": ScalePodSocialNetMitigation,
            # Assign pod to non-existent node
            "assign_to_non_existent_node_social_net-detection-1": AssignNonExistentNodeSocialNetDetection,
            "assign_to_non_existent_node_social_net-localization-1": AssignNonExistentNodeSocialNetLocalization,
            "assign_to_non_existent_node_social_net-analysis-1": AssignNonExistentNodeSocialNetAnalysis,
            "assign_to_non_existent_node_social_net-mitigation-1": AssignNonExistentNodeSocialNetMitigation,
            # Chaos mesh container kill
            "container_kill-detection": ContainerKillDetection,
            "container_kill-localization": ContainerKillLocalization,
            # Pod failure
            "pod_failure_hotel_res-detection-1": PodFailureDetection,
            "pod_failure_hotel_res-localization-1": PodFailureLocalization,
            # Pod kill
            "pod_kill_hotel_res-detection-1": PodKillDetection,
            "pod_kill_hotel_res-localization-1": PodKillLocalization,
            # Network loss
            "network_loss_hotel_res-detection-1": NetworkLossDetection,
            "network_loss_hotel_res-localization-1": NetworkLossLocalization,
            # Network delay
            "network_delay_hotel_res-detection-1": NetworkDelayDetection,
            "network_delay_hotel_res-localization-1": NetworkDelayLocalization,
            # No operation
            "noop_detection_hotel_reservation-1": lambda: NoOpDetection(
                app_name="hotel"
            ),
            "noop_detection_social_network-1": lambda: NoOpDetection(app_name="social"),
            "noop_detection_astronomy_shop-1": lambda: NoOpDetection(app_name="astronomy_shop"),
            # NOTE: This should be getting fixed by the great powers of @jinghao-jia
            # Kernel fault -> https://github.com/xlab-uiuc/agent-ops/pull/10#issuecomment-2468992285
            # There's a bug in chaos mesh regarding this fault, wait for resolution and retest kernel fault
            # "kernel_fault_hotel_reservation-detection-1": KernelFaultDetection,
            # "kernel_fault_hotel_reservation-localization-1": KernelFaultLocalization
            # "disk_woreout-detection-1": DiskWoreoutDetection,
            # "disk_woreout-localization-1": DiskWoreoutLocalization,
            # Open Telemetry Demo (Astronomy Shop) feature flag failures
            "astronomy_shop_ad_service_failure-detection-1": AdServiceFailureDetection,
            "astronomy_shop_ad_service_failure-localization-1": AdServiceFailureLocalization,
            "astronomy_shop_ad_service_high_cpu-detection-1": AdServiceHighCpuDetection,
            "astronomy_shop_ad_service_high_cpu-localization-1": AdServiceHighCpuLocalization,
            "astronomy_shop_ad_service_manual_gc-detection-1": AdServiceManualGcDetection,
            "astronomy_shop_ad_service_manual_gc-localization-1": AdServiceManualGcLocalization,
            "astronomy_shop_cart_service_failure-detection-1": CartServiceFailureDetection,
            "astronomy_shop_cart_service_failure-localization-1": CartServiceFailureLocalization,
            "astronomy_shop_image_slow_load-detection-1": ImageSlowLoadDetection,
            "astronomy_shop_image_slow_load-localization-1": ImageSlowLoadLocalization,
            "astronomy_shop_kafka_queue_problems-detection-1": KafkaQueueProblemsDetection,
            "astronomy_shop_kafka_queue_problems-localization-1": KafkaQueueProblemsLocalization,
            "astronomy_shop_kafka_queue_problems-mitigation-1": KafkaQueueProblemsMitigation,
            "astronomy_shop_loadgenerator_flood_homepage-detection-1": LoadGeneratorFloodHomepageDetection,
            "astronomy_shop_loadgenerator_flood_homepage-localization-1": LoadGeneratorFloodHomepageLocalization,
            "astronomy_shop_payment_service_failure-detection-1": PaymentServiceFailureDetection,
            "astronomy_shop_payment_service_failure-localization-1": PaymentServiceFailureLocalization,
            "astronomy_shop_payment_service_unreachable-detection-1": PaymentServiceUnreachableDetection,
            "astronomy_shop_payment_service_unreachable-localization-1": PaymentServiceUnreachableLocalization,
            "astronomy_shop_product_catalog_service_failure-detection-1": ProductCatalogServiceFailureDetection,
            "astronomy_shop_product_catalog_service_failure-localization-1": ProductCatalogServiceFailureLocalization,
            "astronomy_shop_recommendation_service_cache_failure-detection-1": RecommendationServiceCacheFailureDetection,
            "astronomy_shop_recommendation_service_cache_failure-localization-1": RecommendationServiceCacheFailureLocalization,
            # Redeployment of namespace without deleting the PV
            "redeploy_without_PV-detection-1": RedeployWithoutPVDetection,
            # "redeploy_without_PV-localization-1": RedeployWithoutPVLocalization,
            "redeploy_without_PV-analysis-1": RedeployWithoutPVAnalysis,
            "redeploy_without_PV-mitigation-1": RedeployWithoutPVMitigation,
            # Assign pod to non-existent node
            "wrong_bin_usage-detection-1": WrongBinUsageDetection,
            "wrong_bin_usage-localization-1": WrongBinUsageLocalization,
            "wrong_bin_usage-analysis-1": WrongBinUsageAnalysis,
            "wrong_bin_usage-mitigation-1": WrongBinUsageMitigation,
            # K8S operator misoperation
            # "operator_overload_replicas-detection-1": K8SOperatorOverloadReplicasDetection,
            # "operator_overload_replicas-localization-1": K8SOperatorOverloadReplicasLocalization,
            # "operator_non_existent_storage-detection-1": K8SOperatorNonExistentStorageDetection,
            # "operator_non_existent_storage-localization-1": K8SOperatorNonExistentStorageLocalization,
            # "operator_invalid_affinity_toleration-detection-1": K8SOperatorInvalidAffinityTolerationDetection,
            # "operator_invalid_affinity_toleration-localization-1": K8SOperatorInvalidAffinityTolerationLocalization,
            # "operator_security_context_fault-detection-1": K8SOperatorSecurityContextFaultDetection,
            # "operator_security_context_fault-localization-1": K8SOperatorSecurityContextFaultLocalization,
            # "operator_wrong_update_strategy-detection-1": K8SOperatorWrongUpdateStrategyDetection,
            # "operator_wrong_update_strategy-localization-1": K8SOperatorWrongUpdateStrategyLocalization,
            # Flower
            "flower_node_stop-detection": FlowerNodeStopDetection,
            "flower_model_misconfig-detection": FlowerModelMisconfigDetection,
        }
        self.DOCKER_REGISTRY = [
            "flower_node_stop-detection",
            "flower_model_misconfig-detection",
        ]

    def get_problem_instance(self, problem_id: str):
        if problem_id not in self.PROBLEM_REGISTRY:
            raise ValueError(f"Problem ID {problem_id} not found in registry.")

        return self.PROBLEM_REGISTRY.get(problem_id)()

    def get_problem(self, problem_id: str):
        return self.PROBLEM_REGISTRY.get(problem_id)

    def get_problem_ids(self, task_type: str = None):
        if task_type:
            return [k for k in self.PROBLEM_REGISTRY.keys() if task_type in k]
        return list(self.PROBLEM_REGISTRY.keys())

    def get_problem_count(self, task_type: str = None):
        if task_type:
            return len([k for k in self.PROBLEM_REGISTRY.keys() if task_type in k])
        return len(self.PROBLEM_REGISTRY)
    
    def get_problem_deployment(self, problem_id: str):
        if problem_id in self.DOCKER_REGISTRY:
            return "docker"
        return "k8s"
