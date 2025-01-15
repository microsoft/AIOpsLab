# Copyright (c) Microsoft Corporation
# Licensed under the Apache License, Version 2.0. See LICENSE file in the project root for full license information.

#!/bin/bash

ADMIN_USER="admin"
ADMIN_PWD="admin"

TARGET_DB="rate-db"
# TARGET_DB="geo-db"
READ_WRITE_ROLE="readWrite"

echo "Restoring readWrite privilege to the $ADMIN_USER user for the $TARGET_DB database..."

# Grant readWrite role on the target database
mongo admin -u $ADMIN_USER -p $ADMIN_PWD --authenticationDatabase admin \
     --eval "db.grantRolesToUser('$ADMIN_USER', [{role: '$READ_WRITE_ROLE', db: '$TARGET_DB'}]);"

echo "Privilege restored successfully"