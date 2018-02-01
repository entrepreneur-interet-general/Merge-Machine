#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from merge_machine.analyzers import generate_resources

# =============================================================================
# Define the custom analyzers you want to generate resources from
# =============================================================================

analyzers = ['city', 'country', 'organization']

# =============================================================================
# Generate resources /!\ SUDO PERMISSIONS MIGHT BE REQUIRED
# =============================================================================

generate_resources(analyzers, elasticsearch_resource_dir=None, force=False)
