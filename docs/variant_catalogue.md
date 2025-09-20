# Variant-enabled incident catalogue

This catalogue summarises the incident families that currently support AIOpsLab's
variant execution mode. Each family can be delivered as a sequence of Detection,
Localization, Analysis and Mitigation tasks; the table for every family links to
the static task implementations and their variant-enabled counterparts so that
curriculum authors can quickly inspect the grading logic.

## Kubernetes target port misconfiguration (Social Network)

- **Fault injector**: [`VirtualizationFaultInjector`](../aiopslab/generators/fault/inject_virtual.py)
  with the `misconfig_k8s` scenario to write an incorrect `targetPort` on the
  chosen service.
- **Variant coverage**: [`K8STargetPortMisconfigVariantBase`](../aiopslab/orchestrator/problems/k8s_target_port_misconfig/target_port_variant.py)
  randomises both the affected Social Network service and the wrong port number.
- **Analysis expectations**: the static RCA task expects `system_level="Virtualization"`
  and `fault_type="Misconfiguration"`, while the variant mixin records
  `system_level="Application"`/`fault_type="Misconfiguration"` in its metadata so
  that retries can grade alternative fault scopes.
- **Mitigation checks**: the static task validates that the service's
  `targetPort` has been restored to `9090` and that all pods are healthy; the
  variant task uses the mixin's `pods_ready` expectation for the misconfigured
  service.

| Role | Static task | Variant task | Evaluation focus |
| --- | --- | --- | --- |
| Detection | [`K8STargetPortMisconfigDetection`](../aiopslab/orchestrator/problems/k8s_target_port_misconfig/target_port.py) | [`K8STargetPortMisconfigVariantDetection`](../aiopslab/orchestrator/problems/k8s_target_port_misconfig/target_port_variant.py) | Accepts a case-insensitive `"Yes"` / `"No"` answer. |
| Localization | [`K8STargetPortMisconfigLocalization`](../aiopslab/orchestrator/problems/k8s_target_port_misconfig/target_port.py) | [`K8STargetPortMisconfigVariantLocalization`](../aiopslab/orchestrator/problems/k8s_target_port_misconfig/target_port_variant.py) | Requires an exact or subset match on the faulty service name. |
| Analysis | [`K8STargetPortMisconfigAnalysis`](../aiopslab/orchestrator/problems/k8s_target_port_misconfig/target_port.py) | [`K8STargetPortMisconfigVariantAnalysis`](../aiopslab/orchestrator/problems/k8s_target_port_misconfig/target_port_variant.py) | Checks the submitted `system_level`/`fault_type` pair against the expectations above. |
| Mitigation | [`K8STargetPortMisconfigMitigation`](../aiopslab/orchestrator/problems/k8s_target_port_misconfig/target_port.py) | [`K8STargetPortMisconfigVariantMitigation`](../aiopslab/orchestrator/problems/k8s_target_port_misconfig/target_port_variant.py) | Static task enforces `targetPort=9090` and ready pods; variant task reuses `pods_ready` checks. |

## Hotel Reservation application misconfiguration

- **Fault injector**: [`ApplicationFaultInjector`](../aiopslab/generators/fault/inject_app.py)
  with the `misconfig_app` scenario against the configured microservice.
- **Variant coverage**: [`MisconfigAppVariantBase`](../aiopslab/orchestrator/problems/misconfig_app/misconfig_app_variant.py)
  varies both the target Hotel Reservation service and the misconfiguration type
  (environment, port, connection limits, etc.).
- **Analysis expectations**: static and variant RCA tasks agree on
  `system_level="Application"` and `fault_type="Misconfiguration"`.
- **Mitigation checks**: the static task polls until every pod in the namespace
  is healthy, while the variant task relies on the mixin's namespace-level
  `pods_ready` expectation.

| Role | Static task | Variant task | Evaluation focus |
| --- | --- | --- | --- |
| Detection | [`MisconfigAppHotelResDetection`](../aiopslab/orchestrator/problems/misconfig_app/misconfig_app_hotel_res.py) | [`MisconfigAppVariantDetection`](../aiopslab/orchestrator/problems/misconfig_app/misconfig_app_variant.py) | Case-insensitive `"Yes"` / `"No"` answer. |
| Localization | [`MisconfigAppHotelResLocalization`](../aiopslab/orchestrator/problems/misconfig_app/misconfig_app_hotel_res.py) | [`MisconfigAppVariantLocalization`](../aiopslab/orchestrator/problems/misconfig_app/misconfig_app_variant.py) | Requires the misconfigured service name (exact match or single-item superset). |
| Analysis | [`MisconfigAppHotelResAnalysis`](../aiopslab/orchestrator/problems/misconfig_app/misconfig_app_hotel_res.py) | [`MisconfigAppVariantAnalysis`](../aiopslab/orchestrator/problems/misconfig_app/misconfig_app_variant.py) | Verifies `system_level="Application"` and `fault_type="Misconfiguration"`. |
| Mitigation | [`MisconfigAppHotelResMitigation`](../aiopslab/orchestrator/problems/misconfig_app/misconfig_app_hotel_res.py) | [`MisconfigAppVariantMitigation`](../aiopslab/orchestrator/problems/misconfig_app/misconfig_app_variant.py) | Static task polls for healthy pods; variant task uses namespace `pods_ready`. |

## Social Network scale-to-zero (Kubernetes)

- **Fault injector**: [`VirtualizationFaultInjector`](../aiopslab/generators/fault/inject_virtual.py)
  with `scale_pods_to_zero` against the chosen deployment.
- **Variant coverage**: [`ScalePodVariantBase`](../aiopslab/orchestrator/problems/scale_pod/scale_pod_variant.py)
  selects any Social Network service or backing store to scale to zero replicas.
- **Analysis expectations**: both static and variant RCAs expect
  `system_level="Virtualization"` and `fault_type="Operation Error"`.
- **Mitigation checks**: the static task confirms replicas return to one and
  that pods are ready; the variant metadata adds both `pods_ready` and
  deployment replica checks for the selected service.

| Role | Static task | Variant task | Evaluation focus |
| --- | --- | --- | --- |
| Detection | [`ScalePodSocialNetDetection`](../aiopslab/orchestrator/problems/scale_pod/scale_pod_social_net.py) | [`ScalePodVariantDetection`](../aiopslab/orchestrator/problems/scale_pod/scale_pod_variant.py) | Case-insensitive `"Yes"`. |
| Localization | [`ScalePodSocialNetLocalization`](../aiopslab/orchestrator/problems/scale_pod/scale_pod_social_net.py) | [`ScalePodVariantLocalization`](../aiopslab/orchestrator/problems/scale_pod/scale_pod_variant.py) | Requires the scaled-to-zero deployment. |
| Analysis | [`ScalePodSocialNetAnalysis`](../aiopslab/orchestrator/problems/scale_pod/scale_pod_social_net.py) | [`ScalePodVariantAnalysis`](../aiopslab/orchestrator/problems/scale_pod/scale_pod_variant.py) | Validates `Virtualization` / `Operation Error`. |
| Mitigation | [`ScalePodSocialNetMitigation`](../aiopslab/orchestrator/problems/scale_pod/scale_pod_social_net.py) | [`ScalePodVariantMitigation`](../aiopslab/orchestrator/problems/scale_pod/scale_pod_variant.py) | Static task checks replica counts and pod health; variant uses the mixin's `pods_ready` + deployment checks. |

## Hotel Reservation pod failure

- **Fault injector**: [`SymptomFaultInjector`](../aiopslab/generators/fault/inject_symp.py)
  using the `pod_failure` scenario.
- **Variant coverage**: [`PodFailureVariantBase`](../aiopslab/orchestrator/problems/pod_failure/pod_failure_variant.py)
  rotates the affected Hotel Reservation service.
- **Analysis expectations**: variant RCA expects
  `system_level="Virtualization"` / `fault_type="Operation Error"`.
- **Mitigation checks**: variant mitigation relies on the mixin's `pods_ready`
  expectation; there is no bespoke static mitigation task yet.

| Role | Static task | Variant task | Evaluation focus |
| --- | --- | --- | --- |
| Detection | [`PodFailureDetection`](../aiopslab/orchestrator/problems/pod_failure/pod_failure.py) | [`PodFailureVariantDetection`](../aiopslab/orchestrator/problems/pod_failure/pod_failure_variant.py) | Case-insensitive `"Yes"`. |
| Localization | [`PodFailureLocalization`](../aiopslab/orchestrator/problems/pod_failure/pod_failure.py) | [`PodFailureVariantLocalization`](../aiopslab/orchestrator/problems/pod_failure/pod_failure_variant.py) | Requires the failed service name. |
| Analysis | – | [`PodFailureVariantAnalysis`](../aiopslab/orchestrator/problems/pod_failure/pod_failure_variant.py) | Uses mixin metadata (`Virtualization` / `Operation Error`). |
| Mitigation | – | [`PodFailureVariantMitigation`](../aiopslab/orchestrator/problems/pod_failure/pod_failure_variant.py) | Uses `pods_ready` for the affected service. |

## Hotel Reservation pod kill

- **Fault injector**: [`SymptomFaultInjector`](../aiopslab/generators/fault/inject_symp.py)
  with the `pod_kill` experiment (duration is variant-controlled).
- **Variant coverage**: [`PodKillVariantBase`](../aiopslab/orchestrator/problems/pod_kill/pod_kill_variant.py)
  varies both the target service and the kill duration.
- **Analysis expectations**: mixin metadata records
  `system_level="Virtualization"` / `fault_type="Operation Error"`.
- **Mitigation checks**: variant mitigation uses the `pods_ready`
  expectation for the chosen service.

| Role | Static task | Variant task | Evaluation focus |
| --- | --- | --- | --- |
| Detection | [`PodKillDetection`](../aiopslab/orchestrator/problems/pod_kill/pod_kill.py) | [`PodKillVariantDetection`](../aiopslab/orchestrator/problems/pod_kill/pod_kill_variant.py) | Case-insensitive `"Yes"`. |
| Localization | [`PodKillLocalization`](../aiopslab/orchestrator/problems/pod_kill/pod_kill.py) | [`PodKillVariantLocalization`](../aiopslab/orchestrator/problems/pod_kill/pod_kill_variant.py) | Requires the killed service name. |
| Analysis | – | [`PodKillVariantAnalysis`](../aiopslab/orchestrator/problems/pod_kill/pod_kill_variant.py) | Validates `Virtualization` / `Operation Error`. |
| Mitigation | – | [`PodKillVariantMitigation`](../aiopslab/orchestrator/problems/pod_kill/pod_kill_variant.py) | Ensures pods become ready again. |

## Hotel Reservation network loss

- **Fault injector**: [`SymptomFaultInjector`](../aiopslab/generators/fault/inject_symp.py)
  simulating packet loss via `inject_network_loss`.
- **Variant coverage**: [`NetworkLossVariantBase`](../aiopslab/orchestrator/problems/network_loss/network_loss_variant.py)
  picks the affected service and loss rate.
- **Analysis expectations**: variant RCA expects
  `system_level="Application"` / `fault_type="Network/Storage Issue"`.
- **Mitigation checks**: variant mitigation relies on `pods_ready` for the
  targeted service.

| Role | Static task | Variant task | Evaluation focus |
| --- | --- | --- | --- |
| Detection | [`NetworkLossDetection`](../aiopslab/orchestrator/problems/network_loss/network_loss.py) | [`NetworkLossVariantDetection`](../aiopslab/orchestrator/problems/network_loss/network_loss_variant.py) | Case-insensitive `"Yes"`. |
| Localization | [`NetworkLossLocalization`](../aiopslab/orchestrator/problems/network_loss/network_loss.py) | [`NetworkLossVariantLocalization`](../aiopslab/orchestrator/problems/network_loss/network_loss_variant.py) | Requires the degraded service name. |
| Analysis | – | [`NetworkLossVariantAnalysis`](../aiopslab/orchestrator/problems/network_loss/network_loss_variant.py) | Validates `Application` / `Network/Storage Issue`. |
| Mitigation | – | [`NetworkLossVariantMitigation`](../aiopslab/orchestrator/problems/network_loss/network_loss_variant.py) | Checks that pods for the affected service are ready. |

## Hotel Reservation network delay

- **Fault injector**: [`SymptomFaultInjector`](../aiopslab/generators/fault/inject_symp.py)
  via `inject_network_delay`.
- **Variant coverage**: [`NetworkDelayVariantBase`](../aiopslab/orchestrator/problems/network_delay/network_delay_variant.py)
  rotates both the service and latency budget.
- **Analysis expectations**: variant RCA expects
  `system_level="Application"` / `fault_type="Network/Storage Issue"`.
- **Mitigation checks**: variant mitigation uses `pods_ready` for the delayed
  service.

| Role | Static task | Variant task | Evaluation focus |
| --- | --- | --- | --- |
| Detection | [`NetworkDelayDetection`](../aiopslab/orchestrator/problems/network_delay/network_delay.py) | [`NetworkDelayVariantDetection`](../aiopslab/orchestrator/problems/network_delay/network_delay_variant.py) | Case-insensitive `"Yes"`. |
| Localization | [`NetworkDelayLocalization`](../aiopslab/orchestrator/problems/network_delay/network_delay.py) | [`NetworkDelayVariantLocalization`](../aiopslab/orchestrator/problems/network_delay/network_delay_variant.py) | Requires the impacted service name. |
| Analysis | – | [`NetworkDelayVariantAnalysis`](../aiopslab/orchestrator/problems/network_delay/network_delay_variant.py) | Validates `Application` / `Network/Storage Issue`. |
| Mitigation | – | [`NetworkDelayVariantMitigation`](../aiopslab/orchestrator/problems/network_delay/network_delay_variant.py) | Ensures pods for the service return to ready state. |

## Hotel Reservation container kill (Chaos Mesh)

- **Fault injector**: [`SymptomFaultInjector`](../aiopslab/generators/fault/inject_symp.py)
  drives Chaos Mesh to terminate the selected container.
- **Variant coverage**: [`ContainerKillVariantBase`](../aiopslab/orchestrator/problems/container_kill/container_kill_variant.py)
  enumerates service/container pairs across the application.
- **Analysis expectations**: variant RCA expects
  `system_level="Virtualization"` / `fault_type="Operation Error"`.
- **Mitigation checks**: variant mitigation uses `pods_ready` for the affected
  service.

| Role | Static task | Variant task | Evaluation focus |
| --- | --- | --- | --- |
| Detection | [`ContainerKillDetection`](../aiopslab/orchestrator/problems/container_kill/container_kill.py) | [`ContainerKillVariantDetection`](../aiopslab/orchestrator/problems/container_kill/container_kill_variant.py) | Case-insensitive `"Yes"`. |
| Localization | [`ContainerKillLocalization`](../aiopslab/orchestrator/problems/container_kill/container_kill.py) | [`ContainerKillVariantLocalization`](../aiopslab/orchestrator/problems/container_kill/container_kill_variant.py) | Requires the affected service (static) or service/container (variant metadata). |
| Analysis | – | [`ContainerKillVariantAnalysis`](../aiopslab/orchestrator/problems/container_kill/container_kill_variant.py) | Validates `Virtualization` / `Operation Error`. |
| Mitigation | – | [`ContainerKillVariantMitigation`](../aiopslab/orchestrator/problems/container_kill/container_kill_variant.py) | Uses `pods_ready` checks. |

## TiDB operator misoperation

- **Fault injector**: [`K8SOperatorFaultInjector`](../aiopslab/generators/fault/inject_operator.py)
  applies CR-level mistakes such as replica overloads, invalid tolerations or
  wrong update strategies.
- **Variant coverage**: [`K8SOperatorMisoperationVariantBase`](../aiopslab/orchestrator/problems/operator_misoperation/operator_misoperation_variant.py)
  rotates the specific operator error as well as supporting parameters (replica
  count, toleration effect, storage class, etc.).
- **Analysis expectations**: variant RCA validates
  `system_level="Application"` / `fault_type="Misconfiguration"`.
- **Mitigation checks**: variant mitigation uses the mixin's namespace-level
  `pods_ready` expectation. Static mitigation has not yet been implemented.

| Role | Static task | Variant task | Evaluation focus |
| --- | --- | --- | --- |
| Detection | [`K8SOperatorOverloadReplicasDetection`](../aiopslab/orchestrator/problems/operator_misoperation/overload_replicas.py) | [`K8SOperatorMisoperationVariantDetection`](../aiopslab/orchestrator/problems/operator_misoperation/operator_misoperation_variant.py) | Case-insensitive `"Yes"`. |
| Localization | [`K8SOperatorOverloadReplicasLocalization`](../aiopslab/orchestrator/problems/operator_misoperation/overload_replicas.py) | [`K8SOperatorMisoperationVariantLocalization`](../aiopslab/orchestrator/problems/operator_misoperation/operator_misoperation_variant.py) | Requires the faulty TiDB custom resource (or metadata-derived set). |
| Analysis | – | [`K8SOperatorMisoperationVariantAnalysis`](../aiopslab/orchestrator/problems/operator_misoperation/operator_misoperation_variant.py) | Validates `Application` / `Misconfiguration`. |
| Mitigation | – | [`K8SOperatorMisoperationVariantMitigation`](../aiopslab/orchestrator/problems/operator_misoperation/operator_misoperation_variant.py) | Uses namespace-level `pods_ready` expectations. |

## Other incident families in the registry

Variant mode is layered on top of a much broader pool of static incidents. The
[`ProblemRegistry`](../aiopslab/orchestrator/problems/registry.py) also includes:

- **MongoDB authentication incidents**: credential creation disabled
  (`auth_miss_mongodb-*`), revoked passwords (`revoke_auth_mongodb-*`) and user
  provisioning mistakes (`user_unregistered_mongodb-*`), each with full
  Detection→Mitigation coverage.
- **Stateful service misconfigurations**: redeploying without deleting a PV
  (`redeploy_without_PV-*`) or pointing jobs at the wrong binary image
  (`wrong_bin_usage-*`).
- **Operator task variants**: static TiDB operator scenarios beyond overload
  replicas (invalid tolerations, storage class, update strategy, security
  context) that can be re-enabled as those exercises mature.
- **Chaos experiments without variants (yet)**: container/pod failure, network
  loss/delay and kernel fault prototypes that can still be run in static mode.
- **Application-specific outages**: Astronomy Shop feature-flag incidents,
  Payment Service/Recommendation Service failures, Kafka queue saturation and
  Flower platform experiments (`flower_*`).

Use these entries when you need additional diversity beyond the variant-enabled
families documented above.
