
                (function() {
                    var nodeEnv = typeof require !== 'undefined' && typeof process !== 'undefined';
                    var __module = nodeEnv ? module : {exports:{}};
                    var __filename = 'preview-scripts/assets/3.16小游戏/command_TypeScript/level2/left.js';
                    var __require = nodeEnv ? function (request) {
                        return cc.require(request);
                    } : function (request) {
                        return __quick_compile_project__.require(request, __filename);
                    };
                    function __define (exports, require, module) {
                        if (!nodeEnv) {__quick_compile_project__.registerModule(__filename, module);}"use strict";
cc._RF.push(module, '5ceedp2BGlJTLjvk8lDqHNP', 'left');
// 3.16小游戏/command_TypeScript/level2/left.ts

"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (Object.prototype.hasOwnProperty.call(b, p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
Object.defineProperty(exports, "__esModule", { value: true });
// Learn TypeScript:
//  - https://docs.cocos.com/creator/manual/en/scripting/typescript.html
// Learn Attribute:
//  - https://docs.cocos.com/creator/manual/en/scripting/reference/attributes.html
// Learn life-cycle callbacks:
//  - https://docs.cocos.com/creator/manual/en/scripting/life-cycle-callbacks.html
var main_2 = require("./main");
var _a = cc._decorator, ccclass = _a.ccclass, property = _a.property;
var leftToRight_1 = /** @class */ (function (_super) {
    __extends(leftToRight_1, _super);
    function leftToRight_1() {
        var _this = _super !== null && _super.apply(this, arguments) || this;
        _this.label = null;
        _this.text = 'hello';
        return _this;
    }
    // LIFE-CYCLE CALLBACKS:
    leftToRight_1.prototype.onLoad = function () {
        var manager = cc.director.getCollisionManager();
        manager.enabled = true;
    };
    //产生碰撞会调用
    leftToRight_1.prototype.onCollisionEnter = function (other, self) {
        cc.log(other.node.name);
        if (main_2.default.attack == true) {
            var damage = cc.find("Canvas/bj/kuan/xy/main_damage");
            damage.getComponent(cc.Label).string = "-30";
            var damage2 = cc.find("Canvas/bj/kuan/enemy_damage");
            damage2.getComponent(cc.Label).string = "";
            if (other.node.name == "小鬼") {
                main_2.default.minion1_hp -= 30;
            }
            else if (other.node.name == "小鬼2") {
                main_2.default.minion2_hp -= 30;
            }
            else if (other.node.name == "小鬼3") {
                main_2.default.minion3_hp -= 30;
            }
        }
        main_2.default.attack = false;
    };
    leftToRight_1.prototype.onCollisionStay = function (other) {
        main_2.default.attack = false;
        main_2.default.minion_attack = false;
    };
    leftToRight_1.prototype.onCollisionExit = function (other) {
        if ((main_2.default.minion1_hp <= 0) && (other.node.name == "小鬼")) {
            other.node.active = false;
        }
        else if ((main_2.default.minion2_hp <= 0) && (other.node.name == "小鬼2")) {
            other.node.active = false;
        }
        else if ((main_2.default.minion3_hp <= 0) && (other.node.name == "小鬼3")) {
            other.node.active = false;
            cc.director.loadScene("fight3");
        }
        else {
            if (other.node.name == "小鬼") {
                other.node.children[0].setContentSize(main_2.default.minion1_hp, 19);
            }
            else if (other.node.name == "小鬼2") {
                other.node.children[0].setContentSize(main_2.default.minion2_hp, 19);
            }
            else if (other.node.name == "小鬼3") {
                other.node.children[0].setContentSize(main_2.default.minion3_hp, 19);
            }
        }
    };
    leftToRight_1.prototype.start = function () {
        this.main_x = this.node.position.x;
    };
    leftToRight_1.prototype.update = function (dt) {
        if (main_2.default.attack == true) {
            this.node.setPosition(this.node.position.x + 1000 * dt, this.node.position.y);
        }
        else {
            if (!(this.node.position.x <= this.main_x + 50) && (this.node.position.x >= this.main_x - 50)) {
                this.node.setPosition(this.node.position.x - 1000 * dt, this.node.position.y);
            }
        }
    };
    __decorate([
        property(cc.Label)
    ], leftToRight_1.prototype, "label", void 0);
    __decorate([
        property
    ], leftToRight_1.prototype, "text", void 0);
    leftToRight_1 = __decorate([
        ccclass
    ], leftToRight_1);
    return leftToRight_1;
}(cc.Component));
exports.default = leftToRight_1;

cc._RF.pop();
                    }
                    if (nodeEnv) {
                        __define(__module.exports, __require, __module);
                    }
                    else {
                        __quick_compile_project__.registerModuleFunc(__filename, function () {
                            __define(__module.exports, __require, __module);
                        });
                    }
                })();
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbImFzc2V0c1xcMy4xNuWwj+a4uOaIj1xcY29tbWFuZF9UeXBlU2NyaXB0XFxsZXZlbDJcXGxlZnQudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6Ijs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0FBQUEsb0JBQW9CO0FBQ3BCLHdFQUF3RTtBQUN4RSxtQkFBbUI7QUFDbkIsa0ZBQWtGO0FBQ2xGLDhCQUE4QjtBQUM5QixrRkFBa0Y7QUFDbEYsK0JBQTJCO0FBS3JCLElBQUEsS0FBc0IsRUFBRSxDQUFDLFVBQVUsRUFBbEMsT0FBTyxhQUFBLEVBQUUsUUFBUSxjQUFpQixDQUFDO0FBRzFDO0lBQTJDLGlDQUFZO0lBQXZEO1FBQUEscUVBOEZDO1FBM0ZHLFdBQUssR0FBYSxJQUFJLENBQUM7UUFHdkIsVUFBSSxHQUFXLE9BQU8sQ0FBQzs7SUF3RjNCLENBQUM7SUFsRkcsd0JBQXdCO0lBRXhCLDhCQUFNLEdBQU47UUFDSSxJQUFJLE9BQU8sR0FBRyxFQUFFLENBQUMsUUFBUSxDQUFDLG1CQUFtQixFQUFFLENBQUM7UUFDaEQsT0FBTyxDQUFDLE9BQU8sR0FBRyxJQUFJLENBQUM7SUFDM0IsQ0FBQztJQUdELFNBQVM7SUFDVCx3Q0FBZ0IsR0FBaEIsVUFBaUIsS0FBSyxFQUFDLElBQUk7UUFFeEIsRUFBRSxDQUFDLEdBQUcsQ0FBQyxLQUFLLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxDQUFDO1FBQ3ZCLElBQUcsY0FBTSxDQUFDLE1BQU0sSUFBSSxJQUFJLEVBQUM7WUFDckIsSUFBSSxNQUFNLEdBQUcsRUFBRSxDQUFDLElBQUksQ0FBQywrQkFBK0IsQ0FBQyxDQUFDO1lBQ3RELE1BQU0sQ0FBQyxZQUFZLENBQUMsRUFBRSxDQUFDLEtBQUssQ0FBQyxDQUFDLE1BQU0sR0FBRyxLQUFLLENBQUM7WUFDN0MsSUFBSSxPQUFPLEdBQUcsRUFBRSxDQUFDLElBQUksQ0FBQyw2QkFBNkIsQ0FBQyxDQUFDO1lBQ3JELE9BQU8sQ0FBQyxZQUFZLENBQUMsRUFBRSxDQUFDLEtBQUssQ0FBQyxDQUFDLE1BQU0sR0FBRyxFQUFFLENBQUM7WUFDNUMsSUFBRyxLQUFLLENBQUMsSUFBSSxDQUFDLElBQUksSUFBSSxJQUFJLEVBQUU7Z0JBQzNCLGNBQU0sQ0FBQyxVQUFVLElBQUksRUFBRSxDQUFDO2FBQ3hCO2lCQUFPLElBQUcsS0FBSyxDQUFDLElBQUksQ0FBQyxJQUFJLElBQUksS0FBSyxFQUFFO2dCQUNwQyxjQUFNLENBQUMsVUFBVSxJQUFJLEVBQUUsQ0FBQzthQUN4QjtpQkFBTyxJQUFHLEtBQUssQ0FBQyxJQUFJLENBQUMsSUFBSSxJQUFJLEtBQUssRUFBRTtnQkFDcEMsY0FBTSxDQUFDLFVBQVUsSUFBSSxFQUFFLENBQUM7YUFDeEI7U0FDSDtRQUVELGNBQU0sQ0FBQyxNQUFNLEdBQUcsS0FBSyxDQUFDO0lBQzFCLENBQUM7SUFHRCx1Q0FBZSxHQUFmLFVBQWdCLEtBQUs7UUFDakIsY0FBTSxDQUFDLE1BQU0sR0FBRyxLQUFLLENBQUM7UUFDdEIsY0FBTSxDQUFDLGFBQWEsR0FBRyxLQUFLLENBQUM7SUFDakMsQ0FBQztJQUVELHVDQUFlLEdBQWYsVUFBZ0IsS0FBSztRQU1sQixJQUFHLENBQUMsY0FBTSxDQUFDLFVBQVUsSUFBSSxDQUFDLENBQUMsSUFBRSxDQUFDLEtBQUssQ0FBQyxJQUFJLENBQUMsSUFBSSxJQUFJLElBQUksQ0FBQyxFQUFFO1lBQ3ZELEtBQUssQ0FBQyxJQUFJLENBQUMsTUFBTSxHQUFHLEtBQUssQ0FBQztTQUMxQjthQUFPLElBQUcsQ0FBQyxjQUFNLENBQUMsVUFBVSxJQUFJLENBQUMsQ0FBQyxJQUFFLENBQUMsS0FBSyxDQUFDLElBQUksQ0FBQyxJQUFJLElBQUksS0FBSyxDQUFDLEVBQUU7WUFDaEUsS0FBSyxDQUFDLElBQUksQ0FBQyxNQUFNLEdBQUcsS0FBSyxDQUFDO1NBQzFCO2FBQU0sSUFBRyxDQUFDLGNBQU0sQ0FBQyxVQUFVLElBQUksQ0FBQyxDQUFDLElBQUUsQ0FBQyxLQUFLLENBQUMsSUFBSSxDQUFDLElBQUksSUFBSSxLQUFLLENBQUMsRUFBRTtZQUMvRCxLQUFLLENBQUMsSUFBSSxDQUFDLE1BQU0sR0FBRyxLQUFLLENBQUM7WUFDMUIsRUFBRSxDQUFDLFFBQVEsQ0FBQyxTQUFTLENBQUMsUUFBUSxDQUFDLENBQUM7U0FDaEM7YUFBTTtZQUNILElBQUcsS0FBSyxDQUFDLElBQUksQ0FBQyxJQUFJLElBQUksSUFBSSxFQUFFO2dCQUMzQixLQUFLLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDLENBQUMsQ0FBQyxjQUFjLENBQUMsY0FBTSxDQUFDLFVBQVUsRUFBRSxFQUFFLENBQUMsQ0FBQzthQUM3RDtpQkFBTyxJQUFHLEtBQUssQ0FBQyxJQUFJLENBQUMsSUFBSSxJQUFJLEtBQUssRUFBRTtnQkFDcEMsS0FBSyxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxDQUFDLENBQUMsY0FBYyxDQUFDLGNBQU0sQ0FBQyxVQUFVLEVBQUUsRUFBRSxDQUFDLENBQUM7YUFDN0Q7aUJBQU8sSUFBRyxLQUFLLENBQUMsSUFBSSxDQUFDLElBQUksSUFBSSxLQUFLLEVBQUU7Z0JBQ3BDLEtBQUssQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsQ0FBQyxDQUFDLGNBQWMsQ0FBQyxjQUFNLENBQUMsVUFBVSxFQUFFLEVBQUUsQ0FBQyxDQUFDO2FBQzdEO1NBQ0o7SUFFSixDQUFDO0lBRUQsNkJBQUssR0FBTDtRQUVBLElBQUksQ0FBQyxNQUFNLEdBQUcsSUFBSSxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxDQUFDO0lBR25DLENBQUM7SUFFRCw4QkFBTSxHQUFOLFVBQVEsRUFBRTtRQUNOLElBQUksY0FBTSxDQUFDLE1BQU0sSUFBSyxJQUFJLEVBQUU7WUFDM0IsSUFBSSxDQUFDLElBQUksQ0FBQyxXQUFXLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxHQUFHLElBQUksR0FBQyxFQUFFLEVBQUUsSUFBSSxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxDQUFDLENBQUM7U0FHNUU7YUFBTTtZQUNILElBQUksQ0FBQyxDQUFDLElBQUksQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsSUFBSSxJQUFJLENBQUMsTUFBTSxHQUFHLEVBQUUsQ0FBQyxJQUFJLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxJQUFJLElBQUksQ0FBQyxNQUFNLEdBQUcsRUFBRSxDQUFDLEVBQUk7Z0JBQ2pHLElBQUksQ0FBQyxJQUFJLENBQUMsV0FBVyxDQUFDLElBQUksQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsR0FBRyxJQUFJLEdBQUMsRUFBRSxFQUFFLElBQUksQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsQ0FBQyxDQUFDO2FBQzNFO1NBRUo7SUFHTCxDQUFDO0lBekZEO1FBREMsUUFBUSxDQUFDLEVBQUUsQ0FBQyxLQUFLLENBQUM7Z0RBQ0k7SUFHdkI7UUFEQyxRQUFROytDQUNjO0lBTk4sYUFBYTtRQURqQyxPQUFPO09BQ2EsYUFBYSxDQThGakM7SUFBRCxvQkFBQztDQTlGRCxBQThGQyxDQTlGMEMsRUFBRSxDQUFDLFNBQVMsR0E4RnREO2tCQTlGb0IsYUFBYSIsImZpbGUiOiIiLCJzb3VyY2VSb290IjoiLyIsInNvdXJjZXNDb250ZW50IjpbIi8vIExlYXJuIFR5cGVTY3JpcHQ6XHJcbi8vICAtIGh0dHBzOi8vZG9jcy5jb2Nvcy5jb20vY3JlYXRvci9tYW51YWwvZW4vc2NyaXB0aW5nL3R5cGVzY3JpcHQuaHRtbFxyXG4vLyBMZWFybiBBdHRyaWJ1dGU6XHJcbi8vICAtIGh0dHBzOi8vZG9jcy5jb2Nvcy5jb20vY3JlYXRvci9tYW51YWwvZW4vc2NyaXB0aW5nL3JlZmVyZW5jZS9hdHRyaWJ1dGVzLmh0bWxcclxuLy8gTGVhcm4gbGlmZS1jeWNsZSBjYWxsYmFja3M6XHJcbi8vICAtIGh0dHBzOi8vZG9jcy5jb2Nvcy5jb20vY3JlYXRvci9tYW51YWwvZW4vc2NyaXB0aW5nL2xpZmUtY3ljbGUtY2FsbGJhY2tzLmh0bWxcclxuaW1wb3J0IG1haW5fMSBmcm9tIFwiLi9tYWluXCJcclxuXHJcblxyXG5cclxuXHJcbmNvbnN0IHtjY2NsYXNzLCBwcm9wZXJ0eX0gPSBjYy5fZGVjb3JhdG9yO1xyXG5cclxuQGNjY2xhc3NcclxuZXhwb3J0IGRlZmF1bHQgY2xhc3MgbGVmdFRvUmlnaHRfMSBleHRlbmRzIGNjLkNvbXBvbmVudCB7XHJcblxyXG4gICAgQHByb3BlcnR5KGNjLkxhYmVsKVxyXG4gICAgbGFiZWw6IGNjLkxhYmVsID0gbnVsbDtcclxuXHJcbiAgICBAcHJvcGVydHlcclxuICAgIHRleHQ6IHN0cmluZyA9ICdoZWxsbyc7XHJcbiAgICBtYWluX3g6IG51bWJlcjtcclxuICAgIFxyXG4gICAgcHVibGljIHN0YXRpYyBjdXJyZW50X3g6IG51bWJlcjtcclxuICAgIHB1YmxpYyBzdGF0aWMgY3VycmVudF95OiBudW1iZXI7XHJcblxyXG4gICAgLy8gTElGRS1DWUNMRSBDQUxMQkFDS1M6XHJcblxyXG4gICAgb25Mb2FkICgpIHtcclxuICAgICAgICB2YXIgbWFuYWdlciA9IGNjLmRpcmVjdG9yLmdldENvbGxpc2lvbk1hbmFnZXIoKTtcclxuICAgICAgICBtYW5hZ2VyLmVuYWJsZWQgPSB0cnVlO1xyXG4gICAgfVxyXG4gICAgXHJcblxyXG4gICAgLy/kuqfnlJ/norDmkp7kvJrosIPnlKhcclxuICAgIG9uQ29sbGlzaW9uRW50ZXIob3RoZXIsc2VsZil7XHJcbiAgICAgIFxyXG4gICAgICAgY2MubG9nKG90aGVyLm5vZGUubmFtZSk7XHJcbiAgICAgICAgaWYobWFpbl8xLmF0dGFjayA9PSB0cnVlKXtcclxuICAgICAgICAgICAgbGV0IGRhbWFnZSA9IGNjLmZpbmQoXCJDYW52YXMvYmova3Vhbi94eS9tYWluX2RhbWFnZVwiKTtcclxuICAgICAgICAgICAgZGFtYWdlLmdldENvbXBvbmVudChjYy5MYWJlbCkuc3RyaW5nID0gXCItMzBcIjtcclxuICAgICAgICAgICAgbGV0IGRhbWFnZTIgPSBjYy5maW5kKFwiQ2FudmFzL2JqL2t1YW4vZW5lbXlfZGFtYWdlXCIpO1xyXG4gICAgICAgICAgICBkYW1hZ2UyLmdldENvbXBvbmVudChjYy5MYWJlbCkuc3RyaW5nID0gXCJcIjtcclxuICAgICAgICAgICBpZihvdGhlci5ub2RlLm5hbWUgPT0gXCLlsI/prLxcIikge1xyXG4gICAgICAgICAgICBtYWluXzEubWluaW9uMV9ocCAtPSAzMDtcclxuICAgICAgICAgICB9IGVsc2UgIGlmKG90aGVyLm5vZGUubmFtZSA9PSBcIuWwj+msvDJcIikge1xyXG4gICAgICAgICAgICBtYWluXzEubWluaW9uMl9ocCAtPSAzMDtcclxuICAgICAgICAgICB9IGVsc2UgIGlmKG90aGVyLm5vZGUubmFtZSA9PSBcIuWwj+msvDNcIikge1xyXG4gICAgICAgICAgICBtYWluXzEubWluaW9uM19ocCAtPSAzMDtcclxuICAgICAgICAgICB9XHJcbiAgICAgICAgfVxyXG4gICAgICAgIFxyXG4gICAgICAgIG1haW5fMS5hdHRhY2sgPSBmYWxzZTtcclxuICAgIH1cclxuXHJcblxyXG4gICAgb25Db2xsaXNpb25TdGF5KG90aGVyKSB7XHJcbiAgICAgICAgbWFpbl8xLmF0dGFjayA9IGZhbHNlO1xyXG4gICAgICAgIG1haW5fMS5taW5pb25fYXR0YWNrID0gZmFsc2U7XHJcbiAgICB9XHJcblxyXG4gICAgb25Db2xsaXNpb25FeGl0KG90aGVyKSB7XHJcblxyXG4gICAgICAgXHJcbiAgICAgICBcclxuICAgICBcclxuICAgICAgXHJcbiAgICAgICBpZigobWFpbl8xLm1pbmlvbjFfaHAgPD0gMCkmJihvdGhlci5ub2RlLm5hbWUgPT0gXCLlsI/prLxcIikpIHsgICBcclxuICAgICAgICBvdGhlci5ub2RlLmFjdGl2ZSA9IGZhbHNlO1xyXG4gICAgICAgfSAgZWxzZSBpZigobWFpbl8xLm1pbmlvbjJfaHAgPD0gMCkmJihvdGhlci5ub2RlLm5hbWUgPT0gXCLlsI/prLwyXCIpKSB7ICAgXHJcbiAgICAgICAgb3RoZXIubm9kZS5hY3RpdmUgPSBmYWxzZTtcclxuICAgICAgIH0gZWxzZSBpZigobWFpbl8xLm1pbmlvbjNfaHAgPD0gMCkmJihvdGhlci5ub2RlLm5hbWUgPT0gXCLlsI/prLwzXCIpKSB7ICAgXHJcbiAgICAgICAgb3RoZXIubm9kZS5hY3RpdmUgPSBmYWxzZTtcclxuICAgICAgICBjYy5kaXJlY3Rvci5sb2FkU2NlbmUoXCJmaWdodDNcIik7XHJcbiAgICAgICB9IGVsc2Uge1xyXG4gICAgICAgICAgIGlmKG90aGVyLm5vZGUubmFtZSA9PSBcIuWwj+msvFwiKSB7XHJcbiAgICAgICAgICAgIG90aGVyLm5vZGUuY2hpbGRyZW5bMF0uc2V0Q29udGVudFNpemUobWFpbl8xLm1pbmlvbjFfaHAsIDE5KTtcclxuICAgICAgICAgICB9IGVsc2UgIGlmKG90aGVyLm5vZGUubmFtZSA9PSBcIuWwj+msvDJcIikge1xyXG4gICAgICAgICAgICBvdGhlci5ub2RlLmNoaWxkcmVuWzBdLnNldENvbnRlbnRTaXplKG1haW5fMS5taW5pb24yX2hwLCAxOSk7XHJcbiAgICAgICAgICAgfSBlbHNlICBpZihvdGhlci5ub2RlLm5hbWUgPT0gXCLlsI/prLwzXCIpIHtcclxuICAgICAgICAgICAgb3RoZXIubm9kZS5jaGlsZHJlblswXS5zZXRDb250ZW50U2l6ZShtYWluXzEubWluaW9uM19ocCwgMTkpO1xyXG4gICAgICAgICAgIH1cclxuICAgICAgIH1cclxuICAgICAgIFxyXG4gICAgfVxyXG4gICAgXHJcbiAgICBzdGFydCAoKSB7XHJcbiAgICAgIFxyXG4gICAgdGhpcy5tYWluX3ggPSB0aGlzLm5vZGUucG9zaXRpb24ueDtcclxuICAgICAgIFxyXG4gICAgIFxyXG4gICAgfVxyXG5cclxuICAgIHVwZGF0ZSAoZHQpIHtcclxuICAgICAgICBpZiAobWFpbl8xLmF0dGFjayA9PSAgdHJ1ZSkge1xyXG4gICAgICAgICB0aGlzLm5vZGUuc2V0UG9zaXRpb24odGhpcy5ub2RlLnBvc2l0aW9uLnggKyAxMDAwKmR0LCB0aGlzLm5vZGUucG9zaXRpb24ueSk7XHJcbiAgICAgICAgICAgIFxyXG4gICAgICAgIFxyXG4gICAgICAgIH0gZWxzZSB7XHJcbiAgICAgICAgICAgIGlmICghKHRoaXMubm9kZS5wb3NpdGlvbi54IDw9IHRoaXMubWFpbl94ICsgNTApICYmICh0aGlzLm5vZGUucG9zaXRpb24ueCA+PSB0aGlzLm1haW5feCAtIDUwKSkgICB7XHJcbiAgICAgICAgICAgIHRoaXMubm9kZS5zZXRQb3NpdGlvbih0aGlzLm5vZGUucG9zaXRpb24ueCAtIDEwMDAqZHQsIHRoaXMubm9kZS5wb3NpdGlvbi55KTtcclxuICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICBcclxuICAgICAgICB9XHJcblxyXG4gICAgIFxyXG4gICAgfVxyXG4gICAgXHJcbn1cclxuXHJcblxyXG4iXX0=