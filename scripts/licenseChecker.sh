#!/bin/bash

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

# Packages approved to have UNKNOWN license (manually verified)
APPROVED_UNKNOWN_PACKAGES=(
    "idna"
    # Add more packages here as needed, one per line
    # "another-package"
)

is_approved() {
    echo "$LICENSE_LIST" | grep -q "$1"
}

# Packages that ship PEP 639 metadata report SPDX expressions such as
# "Apache-2.0 OR BSD-2-Clause". OR means either license may be chosen, so one
# approved term is enough; AND means all terms apply, so every term must be approved.
# AND binds tighter than OR, so splitting on OR first is correct. Parenthesized
# expressions pass only if listed verbatim in ApprovedLicenses.txt.
is_expression_approved() {
    local expr="$1" term
    if [[ "$expr" != *" OR "* && "$expr" != *" AND "* ]] || [[ "$expr" == *"("* ]]; then
        is_approved "$expr"
        return
    fi
    if [[ "$expr" == *" OR "* ]]; then
        while IFS= read -r term; do
            is_expression_approved "$term" && return 0
        done < <(echo "$expr" | sed 's/ OR /\n/g')
        return 1
    fi
    if [[ "$expr" == *" AND "* ]]; then
        while IFS= read -r term; do
            is_approved "$term" || { echo "License '$term' (in '$1') is not in the allowed list."; return 1; }
        done < <(echo "$expr" | sed 's/ AND /\n/g')
        return 0
    fi
}

check_licenses() {
    LICENSE_LIST=$(cat ./ApprovedLicenses.txt | tr '\n' '|'| sed 's/|$//')
    pip-licenses --summary > LicenseSummary.txt
    awk '{$1=""; print $0}' ./LicenseSummary.txt | tail -n +2 | sed 's/;/\n/g' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//'| sort -u > ./newLicenseSummary.txt

    while IFS= read -r line || [[ -n "$line" ]]; do
        # Handle UNKNOWN licenses
        if [[ "$line" == "UNKNOWN" ]]; then
            echo "Checking packages with UNKNOWN license..."
            UNKNOWN_PACKAGES=$(pip-licenses --format=csv | grep -i ",UNKNOWN," | cut -d',' -f1)

            for pkg in $UNKNOWN_PACKAGES; do
                # Check if package is in approved list
                if [[ " ${APPROVED_UNKNOWN_PACKAGES[@]} " =~ " ${pkg} " ]]; then
                    echo "✅ $pkg - approved (in known packages list)"
                else
                    echo "❌ $pkg - NOT APPROVED (UNKNOWN license)"
                    echo "License 'UNKNOWN' is not in the allowed list."
                    exit 1
                fi
            done
            continue
        fi

        # Check other licenses
        if ! is_expression_approved "$line"; then
            echo "License '$line' is not in the allowed list."
            exit 1
        fi
    done < ./newLicenseSummary.txt

    if ! grep -q "prohibited-license: Did not find content matching specified patterns" ./scanOutput.txt; then
        echo "Prohibited License Used in Source Code Scan: "
        sed -n '/⚠  prohibited-license:/,/⚠  third-party-license-file:/p' ./scanOutput.txt | sed '1d;$d'| cat
        exit 1
    fi

    echo "License Check complete"
}

check_licenses
