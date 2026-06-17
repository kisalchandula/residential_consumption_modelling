# data_processing/pv/localization.py

"""
PV roof allocation algorithm.

This module allocates installed PV capacity to suitable roof parts.
The spatial matching between households, property polygons, and roofs
is already performed in PostGIS. This class handles only:

1. Calculate roof suitability scores
2. Rank roof candidates per household
3. Select roof parts based on required PV area
4. Distribute PV capacity among selected roofs
"""

import numpy as np


class PVRoofAllocator:

    def __init__(self, roofs):
        """
        Parameters
        ----------
        roofs : GeoDataFrame
            Roof candidates loaded from PostGIS.
        """

        self.roofs = roofs.copy()


    def allocate(self):
        """
        Run complete PV roof allocation workflow.
        """

        # Step 1: Calculate roof suitability scores
        roofs = self.calculate_roof_scores(self.roofs)

        # Step 2: Rank roofs for each household
        roofs = self.rank_roofs(roofs)

        # Step 3: Select roofs until required PV area is reached
        roofs = self.select_roofs(roofs)

        # Step 4: Allocate installed PV capacity to selected roofs
        roofs = self.allocate_capacity(roofs)

        return roofs


    def calculate_roof_scores(self, roofs):
        """
        Calculate PV suitability score based on:
        - orientation
        - roof area
        - roof slope
        - distance penalty
        """

        roofs = roofs.copy()

        roofs["orientation_distance"] = np.minimum(abs(roofs.orientation.fillna(180) - 180), 360 - abs(roofs.orientation.fillna(180) - 180))

        roofs["orientation_in_pv_range"] = roofs.orientation.between(90, 270).astype(int)

        roofs["orientation_score"] = np.where(roofs.roof_type == "1000", 35, np.where(roofs.orientation_in_pv_range == 0, 0, np.maximum(0, 100 - roofs.orientation_distance)))

        roofs["area_score"] = roofs.roof_area.clip(upper=120)

        roofs["slope_score"] = np.where(roofs.roof_type == "1000", 35, np.where(roofs.slope.isna(), 0, np.maximum(0, 100 - abs(roofs.slope - 35) * 3)))

        if "roof_distance" not in roofs.columns:
            roofs["roof_distance"] = 0

        roofs["distance_penalty"] = roofs.roof_distance.clip(upper=50)

        roofs["pitched_pv_bonus"] = np.where((roofs.roof_type != "1000") & (roofs.orientation_in_pv_range == 1), 20, 0)

        roofs["total_score"] = 0.50 * roofs.orientation_score + 0.30 * roofs.area_score + 0.20 * roofs.slope_score - 0.05 * roofs.distance_penalty + roofs.pitched_pv_bonus

        return roofs


    def rank_roofs(self, roofs):
        """
        Rank roof candidates per household based on total score.
        """

        roofs = roofs.sort_values(["an_fid", "total_score", "roof_area"], ascending=[True, False, False])

        roofs["roof_rank"] = roofs.groupby("an_fid").cumcount() + 1

        roofs["candidate_count"] = roofs.groupby("an_fid")["an_fid"].transform("count")

        return roofs


    def select_roofs(self, roofs):
        """
        Select best roofs until enough area is available
        for the installed PV capacity.

        Assumption:
        1 kWp PV requires approximately 6 m² roof area.
        """

        roofs = roofs.copy()

        roofs["required_pv_area"] = roofs.ea_p_pv.fillna(0) * 6.0

        roofs["cumulative_roof_area"] = roofs.groupby("an_fid")["roof_area"].cumsum()

        selected = roofs[roofs.cumulative_roof_area - roofs.roof_area < roofs.required_pv_area]

        return selected


    def allocate_capacity(self, roofs):
        """
        Distribute household PV capacity among selected roofs
        based on roof score and available area.
        """

        roofs = roofs.copy()

        roofs["allocation_weight"] = roofs.total_score * roofs.roof_area

        roofs["total_weight"] = roofs.groupby("an_fid")["allocation_weight"].transform("sum")

        roofs["assigned_pv_capacity"] = np.where(roofs.total_weight > 0, roofs.ea_p_pv * roofs.allocation_weight / roofs.total_weight, 0)

        return roofs