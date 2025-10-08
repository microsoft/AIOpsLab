import os
import re
from typing import Callable, Dict

from aiopslab.orchestrator.problems.k8s_target_port_misconfig import *
from aiopslab.orchestrator.problems.k8s_target_port_misconfig.target_port_variant import (
    K8STargetPortMisconfigVariantDetection,
    K8STargetPortMisconfigVariantLocalization,
)
from aiopslab.orchestrator.problems.auth_miss_mongodb import *
from aiopslab.orchestrator.problems.revoke_auth import *
from aiopslab.orchestrator.problems.storage_user_unregistered import *
from aiopslab.orchestrator.problems.misconfig_app import *
from aiopslab.orchestrator.problems.misconfig_app.misconfig_app_variant import (
    MisconfigAppVariantAnalysis,
    MisconfigAppVariantDetection,
    MisconfigAppVariantLocalization,
    MisconfigAppVariantMitigation,
)
from aiopslab.orchestrator.problems.scale_pod import *
from aiopslab.orchestrator.problems.scale_pod.scale_pod_variant import (
    ScalePodVariantDetection,
    ScalePodVariantLocalization,
)
from aiopslab.orchestrator.problems.assign_non_existent_node import *
from aiopslab.orchestrator.problems.container_kill import *
from aiopslab.orchestrator.problems.container_kill.container_kill_variant import (
    ContainerKillVariantAnalysis,
    ContainerKillVariantDetection,
    ContainerKillVariantLocalization,
    ContainerKillVariantMitigation,
)
from aiopslab.orchestrator.problems.pod_failure import *
from aiopslab.orchestrator.problems.pod_failure.pod_failure_variant import (
    PodFailureVariantDetection,
    PodFailureVariantLocalization,
)
from aiopslab.orchestrator.problems.pod_kill import *
from aiopslab.orchestrator.problems.pod_kill.pod_kill_variant import (
    PodKillVariantAnalysis,
    PodKillVariantDetection,
    PodKillVariantLocalization,
    PodKillVariantMitigation,
)
from aiopslab.orchestrator.problems.network_loss import *
from aiopslab.orchestrator.problems.network_loss.network_loss_variant import (
    NetworkLossVariantAnalysis,
    NetworkLossVariantDetection,
    NetworkLossVariantLocalization,
    NetworkLossVariantMitigation,
)
from aiopslab.orchestrator.problems.network_delay import *
from aiopslab.orchestrator.problems.network_delay.network_delay_variant import (
    NetworkDelayVariantDetection,
    NetworkDelayVariantLocalization,
)
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
from aiopslab.orchestrator.problems.operator_misoperation.operator_misoperation_variant import (
    K8SOperatorMisoperationVariantAnalysis,
    K8SOperatorMisoperationVariantDetection,
    K8SOperatorMisoperationVariantLocalization,
    K8SOperatorMisoperationVariantMitigation,
)
from aiopslab.orchestrator.problems.flower_node_stop import *
from aiopslab.orchestrator.problems.flower_model_misconfig import *


class ProblemRegistry:
    def __init__(self, variant_mode: str | None = None):
        self.variant_mode = self._resolve_variant_mode(variant_mode)
        self._static_registry: Dict[str, Callable[[], object]] = {
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
            "container_kill-detection-1": ContainerKillDetection,
            "container_kill-localization": ContainerKillLocalization,
            "container_kill-localization-1": ContainerKillLocalization,
            "container_kill-analysis": ContainerKillAnalysis,
            "container_kill-analysis-1": ContainerKillAnalysis,
            "container_kill-mitigation": ContainerKillMitigation,
            "container_kill-mitigation-1": ContainerKillMitigation,
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
        self._canonical_map: Dict[str, str] = {}
        self._augment_with_canonical_ids()
        self._variant_registry = self._build_variant_registry()
        self.PROBLEM_REGISTRY = (
            self._variant_registry
            if self.variant_mode == "variant"
            else self._static_registry
        )

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

    def get_canonical_id(self, problem_id: str) -> str:
        """Return the canonical identifier for a problem id."""
        return self._canonical_map.get(
            problem_id,
            self._canonicalize_problem_id(problem_id),
        )

    @property
    def using_variants(self) -> bool:
        """Whether the registry is currently configured for variant constructors."""
        return self.variant_mode == "variant"

    def _resolve_variant_mode(self, override: str | None) -> str:
        """Resolve variant mode from override or environment variables."""
        if override:
            mode = override.lower()
        else:
            env_mode = os.getenv("AIOPSLAB_PROBLEM_VARIANT_MODE")
            if env_mode:
                mode = env_mode.lower()
            else:
                flag = os.getenv("AIOPSLAB_USE_PROBLEM_VARIANTS")
                if flag and flag.lower() in {"1", "true", "yes", "on"}:
                    mode = "variant"
                else:
                    mode = "static"

        if mode not in {"static", "variant"}:
            mode = "static"
        return mode

    def _augment_with_canonical_ids(self) -> None:
        """Add canonical ids (without numeric suffix) for each problem."""
        original_keys = list(self._static_registry.keys())
        for problem_id in original_keys:
            canonical_id = self._canonicalize_problem_id(problem_id)
            self._canonical_map.setdefault(problem_id, canonical_id)
            if canonical_id not in self._static_registry:
                self._static_registry[canonical_id] = self._static_registry[problem_id]
            self._canonical_map.setdefault(canonical_id, canonical_id)

        # Ensure all entries have canonical mapping
        for problem_id in self._static_registry.keys():
            self._canonical_map.setdefault(
                problem_id, self._canonicalize_problem_id(problem_id)
            )

    def _canonicalize_problem_id(self, problem_id: str) -> str:
        """Strip numeric suffixes to compute canonical identifiers."""
        return re.sub(r"-\d+$", "", problem_id)

    def _build_variant_registry(self) -> Dict[str, Callable[[], object]]:
        """Create registry that returns variant constructors when available."""
        variant_registry = dict(self._static_registry)
        variant_registry.update(self._collect_variant_overrides())
        return variant_registry

    def _collect_variant_overrides(self) -> Dict[str, Callable[[], object]]:
        """Return constructors for tasks that support variant execution."""
        overrides: Dict[str, Callable[[], object]] = {
            # K8s target port misconfiguration
            "k8s_target_port-misconfig-detection": lambda: K8STargetPortMisconfigVariantDetection(
                enable_variants=True
            ),
            "k8s_target_port-misconfig-detection-1": lambda: K8STargetPortMisconfigVariantDetection(
                faulty_service="user-service", enable_variants=True
            ),
            "k8s_target_port-misconfig-detection-2": lambda: K8STargetPortMisconfigVariantDetection(
                faulty_service="text-service", enable_variants=True
            ),
            "k8s_target_port-misconfig-detection-3": lambda: K8STargetPortMisconfigVariantDetection(
                faulty_service="post-storage-service", enable_variants=True
            ),
            "k8s_target_port-misconfig-localization": lambda: K8STargetPortMisconfigVariantLocalization(
                enable_variants=True
            ),
            "k8s_target_port-misconfig-localization-1": lambda: K8STargetPortMisconfigVariantLocalization(
                faulty_service="user-service", enable_variants=True
            ),
            "k8s_target_port-misconfig-localization-2": lambda: K8STargetPortMisconfigVariantLocalization(
                faulty_service="text-service", enable_variants=True
            ),
            "k8s_target_port-misconfig-localization-3": lambda: K8STargetPortMisconfigVariantLocalization(
                faulty_service="post-storage-service", enable_variants=True
            ),
            # Misconfig app (Hotel Reservation)
            "misconfig_app_hotel_res-detection": lambda: MisconfigAppVariantDetection(
                enable_variants=True
            ),
            "misconfig_app_hotel_res-detection-1": lambda: MisconfigAppVariantDetection(
                faulty_service="geo", config_type="env", enable_variants=True
            ),
            "misconfig_app_hotel_res-localization": lambda: MisconfigAppVariantLocalization(
                enable_variants=True
            ),
            "misconfig_app_hotel_res-localization-1": lambda: MisconfigAppVariantLocalization(
                faulty_service="geo", config_type="env", enable_variants=True
            ),
            "misconfig_app_hotel_res-analysis": lambda: MisconfigAppVariantAnalysis(
                enable_variants=True
            ),
            "misconfig_app_hotel_res-analysis-1": lambda: MisconfigAppVariantAnalysis(
                faulty_service="geo", config_type="env", enable_variants=True
            ),
            "misconfig_app_hotel_res-mitigation": lambda: MisconfigAppVariantMitigation(
                enable_variants=True
            ),
            "misconfig_app_hotel_res-mitigation-1": lambda: MisconfigAppVariantMitigation(
                faulty_service="geo", config_type="env", enable_variants=True
            ),
            # Scale pod (Social Network)
            "scale_pod_zero_social_net-detection": lambda: ScalePodVariantDetection(
                enable_variants=True
            ),
            "scale_pod_zero_social_net-detection-1": lambda: ScalePodVariantDetection(
                faulty_service="user-service", enable_variants=True
            ),
            "scale_pod_zero_social_net-localization": lambda: ScalePodVariantLocalization(
                enable_variants=True
            ),
            "scale_pod_zero_social_net-localization-1": lambda: ScalePodVariantLocalization(
                faulty_service="user-service", enable_variants=True
            ),
            # Pod failure (Hotel Reservation)
            "pod_failure_hotel_res-detection": lambda: PodFailureVariantDetection(
                enable_variants=True
            ),
            "pod_failure_hotel_res-detection-1": lambda: PodFailureVariantDetection(
                faulty_service="user", enable_variants=True
            ),
            "pod_failure_hotel_res-localization": lambda: PodFailureVariantLocalization(
                enable_variants=True
            ),
            "pod_failure_hotel_res-localization-1": lambda: PodFailureVariantLocalization(
                faulty_service="user", enable_variants=True
            ),
            # Pod kill (Hotel Reservation)
            "pod_kill_hotel_res-detection": lambda: PodKillVariantDetection(
                enable_variants=True
            ),
            "pod_kill_hotel_res-detection-1": lambda: PodKillVariantDetection(
                faulty_service="user", duration="100s", enable_variants=True
            ),
            "pod_kill_hotel_res-localization": lambda: PodKillVariantLocalization(
                enable_variants=True
            ),
            "pod_kill_hotel_res-localization-1": lambda: PodKillVariantLocalization(
                faulty_service="user", duration="100s", enable_variants=True
            ),
            "pod_kill_hotel_res-analysis": lambda: PodKillVariantAnalysis(
                enable_variants=True
            ),
            "pod_kill_hotel_res-analysis-1": lambda: PodKillVariantAnalysis(
                faulty_service="user", duration="100s", enable_variants=True
            ),
            "pod_kill_hotel_res-mitigation": lambda: PodKillVariantMitigation(
                enable_variants=True
            ),
            "pod_kill_hotel_res-mitigation-1": lambda: PodKillVariantMitigation(
                faulty_service="user", duration="100s", enable_variants=True
            ),
            # Network loss (Hotel Reservation)
            "network_loss_hotel_res-detection": lambda: NetworkLossVariantDetection(
                enable_variants=True
            ),
            "network_loss_hotel_res-detection-1": lambda: NetworkLossVariantDetection(
                faulty_service="user", loss_rate=0.1, enable_variants=True
            ),
            "network_loss_hotel_res-localization": lambda: NetworkLossVariantLocalization(
                enable_variants=True
            ),
            "network_loss_hotel_res-localization-1": lambda: NetworkLossVariantLocalization(
                faulty_service="user", loss_rate=0.1, enable_variants=True
            ),
            "network_loss_hotel_res-analysis": lambda: NetworkLossVariantAnalysis(
                enable_variants=True
            ),
            "network_loss_hotel_res-analysis-1": lambda: NetworkLossVariantAnalysis(
                faulty_service="user", loss_rate=0.1, enable_variants=True
            ),
            "network_loss_hotel_res-mitigation": lambda: NetworkLossVariantMitigation(
                enable_variants=True
            ),
            "network_loss_hotel_res-mitigation-1": lambda: NetworkLossVariantMitigation(
                faulty_service="user", loss_rate=0.1, enable_variants=True
            ),
            # Network delay (Hotel Reservation)
            "network_delay_hotel_res-detection": lambda: NetworkDelayVariantDetection(
                enable_variants=True
            ),
            "network_delay_hotel_res-detection-1": lambda: NetworkDelayVariantDetection(
                faulty_service="user", delay_ms=100, enable_variants=True
            ),
            "network_delay_hotel_res-localization": lambda: NetworkDelayVariantLocalization(
                enable_variants=True
            ),
            "network_delay_hotel_res-localization-1": lambda: NetworkDelayVariantLocalization(
                faulty_service="user", delay_ms=100, enable_variants=True
            ),
            # Container kill (Hotel Reservation)
            "container_kill-detection": lambda: ContainerKillVariantDetection(
                faulty_service="geo",
                faulty_container="hotel-reserv-geo",
                enable_variants=True,
            ),
            "container_kill-detection-1": lambda: ContainerKillVariantDetection(
                faulty_service="geo",
                faulty_container="hotel-reserv-geo",
                enable_variants=True,
            ),
            "container_kill-localization": lambda: ContainerKillVariantLocalization(
                faulty_service="geo",
                faulty_container="hotel-reserv-geo",
                enable_variants=True,
            ),
            "container_kill-localization-1": lambda: ContainerKillVariantLocalization(
                faulty_service="geo",
                faulty_container="hotel-reserv-geo",
                enable_variants=True,
            ),
            "container_kill-analysis": lambda: ContainerKillVariantAnalysis(
                faulty_service="geo",
                faulty_container="hotel-reserv-geo",
                enable_variants=True,
            ),
            "container_kill-analysis-1": lambda: ContainerKillVariantAnalysis(
                faulty_service="geo",
                faulty_container="hotel-reserv-geo",
                enable_variants=True,
            ),
            "container_kill-mitigation": lambda: ContainerKillVariantMitigation(
                faulty_service="geo",
                faulty_container="hotel-reserv-geo",
                enable_variants=True,
            ),
            "container_kill-mitigation-1": lambda: ContainerKillVariantMitigation(
                faulty_service="geo",
                faulty_container="hotel-reserv-geo",
                enable_variants=True,
            ),
        }

        # Operator misoperation variants (if enabled in the registry)
        if "operator_overload_replicas-detection-1" in self._static_registry:
            overrides.update(
                {
                    "operator_overload_replicas-detection": lambda: K8SOperatorMisoperationVariantDetection(
                        fault_type="overload_replicas", enable_variants=True
                    ),
                    "operator_overload_replicas-detection-1": lambda: K8SOperatorMisoperationVariantDetection(
                        fault_type="overload_replicas", enable_variants=True
                    ),
                    "operator_overload_replicas-localization": lambda: K8SOperatorMisoperationVariantLocalization(
                        fault_type="overload_replicas", enable_variants=True
                    ),
                    "operator_overload_replicas-localization-1": lambda: K8SOperatorMisoperationVariantLocalization(
                        fault_type="overload_replicas", enable_variants=True
                    ),
                    "operator_overload_replicas-analysis": lambda: K8SOperatorMisoperationVariantAnalysis(
                        fault_type="overload_replicas", enable_variants=True
                    ),
                    "operator_overload_replicas-mitigation": lambda: K8SOperatorMisoperationVariantMitigation(
                        fault_type="overload_replicas", enable_variants=True
                    ),
                }
            )

        return overrides
